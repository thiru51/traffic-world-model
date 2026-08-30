# Base tag deliberately matches the torch/CUDA pin in pixi.toml (torch 2.6.0 + cu124).
# If you bump torch, bump this tag in the same commit or the container will silently
# link against a different CUDA runtime than the one the pixi env was tested on.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    SDL_VIDEODRIVER=dummy \
    MPLBACKEND=Agg

# MetaDrive draws its top-down observations through pygame and pulls in opencv; both
# need these X/GL shared objects present even though nothing is ever displayed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libx11-6 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# torch is already in the base image; installing it again would pull a second copy.
RUN grep -v '^torch' /app/requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt

COPY src /app/src
COPY configs /app/configs

# MetaDrive fetches its asset pack on first use; doing it at build time keeps the
# container usable without network access.
RUN python -c "from metadrive.engine.asset_loader import AssetLoader; import metadrive; print(metadrive.__version__)" \
    && python -m metadrive.pull_asset || echo "asset pull skipped; will happen at first run"

CMD ["python", "-m", "twm.train"]
