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

# Copy feature extraction sources and pretrained checkpoint
COPY --chown=user:user extract_feat_LP.py /opt/app/extract_feat_LP.py
COPY --chown=user:user extract_feat_LP.sh /opt/app/extract_feat_LP.sh
COPY --chown=user:user requirements_docker.txt /opt/app/requirements_docker.txt
COPY --chown=user:user ./checkpoints/VoCo_L_SSL_head.pt /opt/app/checkpoints/VoCo_L_SSL_head.pt

# Set working directory for installation
WORKDIR /opt/app/

# Install python dependencies for feature extraction
RUN pip install --user -r requirements_docker.txt

# Make entrypoint script executable
RUN chmod +x /opt/app/extract_feat_LP.sh

WORKDIR /opt/app/

ENTRYPOINT []
