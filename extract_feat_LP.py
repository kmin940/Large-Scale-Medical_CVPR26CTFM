"""
extract_feat.py

Extract features from VoCo/VoComni SwinViT backbone for linear probing.
Produces per-sample .h5 files with key 'y_hat' containing 1D feature vectors.

Multi-scale pool+concat strategy:
    hidden_states = swinViT(image)  # 5 hidden states
    features = concat([adaptive_avg_pool3d(hs, 1) for hs in hidden_states])
    # channels: [fs, 2*fs, 4*fs, 8*fs, 16*fs]
    # Base (fs=48): 48+96+192+384+768 = 1488 dims

Preprocessing matches VoCo pretraining:
    ScaleIntensityRanged(a_min=-175, a_max=250, b_min=0, b_max=1, clip=True)
"""

import argparse
import os
from typing import Dict, Hashable, List, Mapping, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Vendored MaskCenterCropd (from CT-NEXUS, numpy-only, no nnssl dependency)
# ---------------------------------------------------------------------------

class MaskCenterCropd(MapTransform):
    """Crop around the centroid of foreground labels in a mask, with zero-padding."""

    def __init__(self, keys, mask_key="mask", roi_size=(384, 384, 384), fg_labels=None):
        super().__init__(keys)
        self.mask_key = mask_key
        self.roi_size = roi_size
        self.fg_labels = fg_labels
        self.img_key = "image"

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        mask_arr = d[self.mask_key]
        # Handle channel dimension
        if isinstance(mask_arr, torch.Tensor):
            mask_np = mask_arr.numpy()
        else:
            mask_np = mask_arr
        if mask_np.ndim == 4:
            mask_np = mask_np[0]

        if self.fg_labels is not None:
            mask_binary = np.isin(mask_np, self.fg_labels).astype(np.uint8)
            coords = np.argwhere(mask_binary == 1)

            if coords.size == 0:
                # Fallback: center of volume
                center = (mask_np.shape[0] // 2, mask_np.shape[1] // 2, mask_np.shape[2] // 2)
            else:
                center = tuple(coords.mean(axis=0).astype(int))
        else:
            img_arr = d[self.img_key]
            if isinstance(img_arr, torch.Tensor):
                img_arr = img_arr.numpy()
            shape_img = img_arr.shape[1:] if img_arr.ndim == 4 else img_arr.shape
            center = (shape_img[0] // 2, shape_img[1] // 2, shape_img[2] // 2)

        for key in self.keys:
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)
            if is_tensor:
                arr = arr.numpy()
            has_channel = arr.ndim == 4
            arr_data = arr[0] if has_channel else arr
            cropped = self._crop_with_padding(arr_data, center, self.roi_size)
            if has_channel:
                cropped = cropped[np.newaxis, ...]
            d[key] = torch.from_numpy(cropped) if is_tensor else cropped

        return d

    @staticmethod
    def _crop_with_padding(arr, center, size):
        zc, yc, xc = center
        dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2

        z_start, z_end = zc - dz, zc + dz
        y_start, y_end = yc - dy, yc + dy
        x_start, x_end = xc - dx, xc + dx

        cropped = np.zeros(size, dtype=arr.dtype)

        z_s, z_e = max(z_start, 0), min(z_end, arr.shape[0])
        y_s, y_e = max(y_start, 0), min(y_end, arr.shape[1])
        x_s, x_e = max(x_start, 0), min(x_end, arr.shape[2])

        zo, yo, xo = z_s - z_start, y_s - y_start, x_s - x_start

        cropped[
            zo : zo + (z_e - z_s),
            yo : yo + (y_e - y_s),
            xo : xo + (x_e - x_s),
        ] = arr[z_s:z_e, y_s:y_e, x_s:x_e]

        return cropped


# ---------------------------------------------------------------------------
# Preprocessing pipelines
# ---------------------------------------------------------------------------

def build_non_roi_transforms(roi_size, spacing):
    """Non-ROI diseases: no mask cropping. roi_size is (x, y, z) in RAS order."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=list(roi_size)),
        ToTensord(keys=["image"]),
    ])


def build_roi_transforms(roi_size, spacing, fg_labels=None):
    """ROI diseases: mask-guided center crop. roi_size is (x, y, z) in RAS order."""
    if fg_labels is None:
        fg_labels = [1]
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"], pixdim=spacing, mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        MaskCenterCropd(
            keys=["image", "mask"],
            mask_key="mask",
            roi_size=tuple(roi_size),
            fg_labels=fg_labels,
        ),
        ToTensord(keys=["image"]),
    ])


# ---------------------------------------------------------------------------
# Model loading (from Downstream/monai/CC-CCII/utils/utils.py)
# ---------------------------------------------------------------------------

def load_pretrained_weights(model, model_dict):
    """Load pretrained weights with flexible state_dict extraction and key fixing."""
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        print("Tag 'backbone.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()
    new_state_dict = {
        k: state_dict[k]
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size())
        else current_model_dict[k]
        for k in current_model_dict.keys()
    }

    model.load_state_dict(new_state_dict, strict=True)
    print("Using VoCo pretrained backbone weights !!!!!!!")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract SwinViT features for linear probing")
    parser.add_argument("-i", "--input", required=True, help="Directory with .nii.gz images")
    parser.add_argument("-o", "--output", required=True, help="Output directory for .h5 features")
    parser.add_argument("--masks_path", default=None, help="Optional mask directory for ROI diseases")
    parser.add_argument("--checkpoint", required=True, help="Path to VoCo/VoComni .pt file")
    parser.add_argument("--feature_size", type=int, default=48, choices=[48, 96, 192],
                        help="SwinViT embed_dim: 48 (B), 96 (L), 192 (H)")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[336, 336, 320],
                        help="Spatial crop size as x y z in RAS order (default: 336 336 320)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                        help="Target spacing (default: 1.5 1.5 1.5)")
    parser.add_argument("--fg_labels", type=int, nargs="+", default=None,
                        help="Foreground label IDs for mask cropping (default: [1])")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacing = tuple(args.spacing)

    # Build SwinViT backbone
    fs = args.feature_size
    swinViT = SwinViT(
        in_chans=1,
        embed_dim=fs,
        window_size=(7, 7, 7),
        patch_size=(2, 2, 2),
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=torch.nn.LayerNorm,
        use_checkpoint=False,
        spatial_dims=3,
        use_v2=True,
    )

    # Load pretrained weights
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    swinViT = load_pretrained_weights(swinViT, ckpt)
    swinViT.eval()
    swinViT.to(device)

    # Feature dimension: sum of all hidden state channels
    embed_dim = fs + 2 * fs + 4 * fs + 8 * fs + 16 * fs  # 31 * fs
    print(f"Feature dimension: {embed_dim} (feature_size={fs})")

    # Gather input files
    image_files = sorted([
        f for f in os.listdir(args.input)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ])
    print(f"Found {len(image_files)} images in {args.input}")

    # Build transforms
    use_masks = args.masks_path is not None and os.path.isdir(args.masks_path)
    fg_labels = args.fg_labels if args.fg_labels else [1]

    if use_masks:
        print(f"Using ROI transforms with masks from {args.masks_path}, fg_labels={fg_labels}")
        transforms = build_roi_transforms(args.roi_size, spacing, fg_labels=fg_labels)
    else:
        print("Using non-ROI transforms (no masks)")
        transforms = build_non_roi_transforms(args.roi_size, spacing)

    # Extract features
    skipped = []
    for filename in tqdm(image_files, desc="Extracting features"):
        basename = filename.replace(".nii.gz", "").replace(".nii", "")
        out_path = os.path.join(args.output, basename + ".h5")

        if os.path.exists(out_path):
            continue

        image_path = os.path.join(args.input, filename)

        try:
            if use_masks:
                mask_filename = filename  # assume same filename in masks_path
                mask_path = os.path.join(args.masks_path, mask_filename)
                if not os.path.exists(mask_path):
                    print(f"Warning: mask not found for {filename}, skipping")
                    skipped.append(filename)
                    continue
                data = {"image": image_path, "mask": mask_path}
            else:
                data = {"image": image_path}

            transformed = transforms(data)
            image_tensor = transformed["image"].unsqueeze(0).to(device)
            # save in nii.gz using nibabel
            #import nibabel as nib
            #print("SHAPE", image_tensor.squeeze(0).squeeze(0).cpu().numpy().shape)
            #nib.save(nib.Nifti1Image(image_tensor.squeeze(0).squeeze(0).cpu().numpy(), np.eye(4)), '/home/jma/Documents/cryoSumin/CT_FM/CT-agent/Large-Scale-Medical/dump/test.nii.gz')
            #exit()

            with torch.no_grad():
                hidden_states = swinViT(image_tensor)
                features = torch.cat(
                    [F.adaptive_avg_pool3d(hs, 1).flatten(1) for hs in hidden_states],
                    dim=1,
                )
                features = features.squeeze(0).cpu().numpy()

            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("y_hat", data=features)

            del image_tensor, hidden_states
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped.append(filename)
            continue

    print(f"\nDone! Extracted features for {len(image_files) - len(skipped)}/{len(image_files)} images.")
    if skipped:
        print(f"Skipped {len(skipped)} files: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")


if __name__ == "__main__":
    main()
