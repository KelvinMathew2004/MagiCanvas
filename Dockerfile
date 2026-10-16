FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# System libs some renderer backends (e.g. skia) need at runtime, plus git
# (build-time only) so pip can install cr-renderer straight from its repo.
# Harmless to include even if your engine doesn't end up using them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch first (from the PyTorch CPU wheel index) so the large default
# CUDA build is never pulled in; then the rest of the engine's dependencies.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

# Bake the SigLIP perceptual re-ranker weights into the image at BUILD time, so
# engine/vision.py can load them offline. This is the ONLY step that needs the
# network, and it runs at build time only -- stage-one scoring runs with
# --network none, where these baked weights are read with local_files_only.
COPY scripts/fetch_siglip.py ./scripts/fetch_siglip.py
RUN python scripts/fetch_siglip.py

# If you add dependencies for your own resize() approach, add them to
# requirements.txt above — everything needed must install at BUILD time.
# Stage-one scoring runs with --network none, so nothing at runtime can
# reach out to the internet (no pip installs, no API calls, no model
# downloads at run time — pull those in during the build instead).

COPY chameleon_design_schema_v0.9.json .
COPY brand_kit/ ./brand_kit/
COPY renderer/ ./renderer/
COPY engine/ ./engine/

# input/ and output/ are mounted as volumes at run time (see README) —
# not baked into the image.
RUN mkdir -p /app/input /app/output

ENTRYPOINT ["python", "-m", "engine.main"]
