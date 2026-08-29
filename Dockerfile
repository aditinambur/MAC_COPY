# Use a lightweight PyTorch base image with CUDA support
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install system dependencies for GUI/rendering libraries (needed by Gym/MPE)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy requirements files
COPY requirements.txt .
COPY api/requirements_api.txt ./api/

# Install basic dependencies.
# - Pin numpy<2 to avoid breaking changes in NumPy 2.0 (e.g., AttributeError: np.float_)
# - Pin wandb<0.17 to avoid Go compiler requirements
# - Pin protobuf<=3.20.3 to prevent descriptor check errors
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV WANDB_MODE=disabled
ENV WANDB_SILENT=true

RUN pip install --no-cache-dir \
    "protobuf<=3.20.3" \
    "numpy<2" \
    gym==0.19.0 \
    pygame \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    absl-py \
    tensorboard \
    tensorboardX \
    "wandb<0.17" \
    setproctitle \
    imageio \
    Pillow \
    "pyglet<=1.5.27"

# Install FastAPI backend API requirements
RUN pip install --no-cache-dir -r api/requirements_api.txt

# Copy the rest of the codebase
COPY . .

# Install the repository in editable mode
RUN pip install -e . --no-deps

# Expose the API port
EXPOSE 8000

# Start FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]