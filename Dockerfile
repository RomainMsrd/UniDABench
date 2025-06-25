# Use the NVIDIA CUDA 12.1 base image (Ubuntu 22.04)
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Install system dependencies (including git for later cloning)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Update PATH so conda is available
ENV PATH=$CONDA_DIR/bin:$PATH

# Create a working directory
WORKDIR /workspace

# Create a new temporary directory
RUN mkdir -p /workspace/tmp

# Set the TMPDIR environment variable to use this directory
ENV TMPDIR=/workspace/tmp

# Set default shell to bash for subsequent RUN commands
SHELL ["/bin/bash", "-c"]

# Copy the environment.yml file into the container
COPY environment.yml /workspace/environment.yml

# Create the Conda environment from environment.yml
RUN conda env create -f /workspace/environment.yml

# Verify that the environment is working (optional)
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate adatime && \
    python --version

# Install additional packages into the "adatime" environment via pip
RUN conda run -n adatime pip install torch torchvision torchaudio
RUN conda run -n adatime pip install s3-timeseries
RUN conda run -n adatime pip install timm

# Append the Conda initialization and environment activation command to .bashrc so it happens automatically
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate adatime" >> /root/.bashrc

# Set default command to open a shell (which will process .bashrc)
CMD ["/bin/bash"]

