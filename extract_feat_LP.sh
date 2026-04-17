#!/bin/bash
set -e

# Default paths for Docker environment
INPUT_DIR="${INPUT_DIR:-/workspace/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
MASKS_DIR="${MASKS_DIR:-}"                          # Optional masks directory (ROI diseases)
CHECKPOINT="${CHECKPOINT:-./checkpoints/VoCo_L_SSL_head.pt}"

# Model / preprocessing config
FEATURE_SIZE="${FEATURE_SIZE:-96}"                 # 48 (B), 96 (L), 192 (H)
ROI_SIZE_XY="${ROI_SIZE_XY:-224}"                   # x, y in RAS
ROI_SIZE_Z="${ROI_SIZE_Z:-320}"                     # z in RAS
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SPACING_X="${SPACING_X:-1.5}"
SPACING_Y="${SPACING_Y:-1.5}"
SPACING_Z="${SPACING_Z:-1.5}"
FG_LABELS="${FG_LABELS:-}"                          # Space-separated label ids, e.g. "1 2"

# Build command
CMD="python extract_feat_LP.py \
  -i \"$INPUT_DIR\" \
  -o \"$OUTPUT_DIR\" \
  --checkpoint \"$CHECKPOINT\" \
  --feature_size $FEATURE_SIZE \
  --roi_size $ROI_SIZE_XY $ROI_SIZE_XY $ROI_SIZE_Z \
  --spacing $SPACING_X $SPACING_Y $SPACING_Z \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS"

# Add masks_path for ROI diseases
if [ -n "$MASKS_DIR" ]; then
    CMD="$CMD --masks_path \"$MASKS_DIR\""
fi

# # Add fg_labels if set
# if [ -n "$FG_LABELS" ]; then
#     CMD="$CMD --fg_labels $FG_LABELS"
# fi

# Run feature extraction
eval $CMD
