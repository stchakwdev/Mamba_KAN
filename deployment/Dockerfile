FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    gcc \
    g++ \
    curl \
    wget \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install Python development headers
RUN apt-get update && apt-get install -y python3-dev && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (force reinstall to ensure CUDA compatibility)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install core ML libraries
RUN pip install \
    pytorch-lightning>=2.0.0 \
    transformers>=4.30.0 \
    datasets>=2.14.0 \
    evaluate>=0.4.0 \
    wandb>=0.15.0 \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn

# Install KAN dependencies
RUN pip install pykan>=0.2.0
RUN pip install git+https://github.com/Blealtan/efficient-kan.git

# Install Mamba SSM with CUDA support
RUN pip install mamba-ssm>=1.2.0

# Install development tools
RUN pip install pytest black isort jupyterlab

# Install RunPod utilities
RUN pip install python-dotenv requests

# Set up Jupyter
RUN pip install notebook ipywidgets

# Expose ports
EXPOSE 8888 22

# Set default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]