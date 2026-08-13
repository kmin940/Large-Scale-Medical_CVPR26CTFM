FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
# Time zone setting
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 tzdata \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/inputs /workspace/outputs \
    && chown user:user /workspace/inputs /workspace/outputs

USER user

ENV PATH="/home/user/.local/bin:${PATH}"

# Upgrade pip
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# Which VoCo backbone to ship: B (feature_size 48), L (96) or H (192).
# Defaults reproduce the original VoCo-L image byte-for-byte in behaviour.
#   docker build -t voco_lp .                                        # VoCo-L
#   docker build --build-arg VOCO_VARIANT=B --build-arg FEATURE_SIZE=48 -t voco-b_lp .
ARG VOCO_VARIANT=L
ARG FEATURE_SIZE=96

# Copy feature extraction sources and pretrained checkpoint
COPY --chown=user:user extract_feat_LP.py /opt/app/extract_feat_LP.py
COPY --chown=user:user extract_feat_LP.sh /opt/app/extract_feat_LP.sh
COPY --chown=user:user requirements_docker.txt /opt/app/requirements_docker.txt
COPY --chown=user:user ./checkpoints/VoCo_${VOCO_VARIANT}_SSL_head.pt /opt/app/checkpoints/VoCo_${VOCO_VARIANT}_SSL_head.pt

# extract_feat_LP.sh already reads both from the environment; setting them here means
# the image needs no extra -e flags at run time.
ENV CHECKPOINT=./checkpoints/VoCo_${VOCO_VARIANT}_SSL_head.pt
ENV FEATURE_SIZE=${FEATURE_SIZE}

# Set working directory for installation
WORKDIR /opt/app/

# Install python dependencies for feature extraction
RUN pip install --user -r requirements_docker.txt

# Make entrypoint script executable
RUN chmod +x /opt/app/extract_feat_LP.sh

WORKDIR /opt/app/

ENTRYPOINT []
