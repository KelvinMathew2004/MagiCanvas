FROM --platform=linux/amd64 python:3.11-slim AS runtime

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
# engine/vision.py can load them offline. Model download steps need network only
# during the image build; stage-one scoring runs with --network none, where
# these baked weights are read with local_files_only.
COPY scripts/fetch_siglip.py ./scripts/fetch_siglip.py
RUN python scripts/fetch_siglip.py

# Qwen is enabled with the local offline weights below. Its local
# weights are included so a judging container can run entirely offline.
COPY scripts/fetch_qwen.py ./scripts/fetch_qwen.py
ARG QWEN_MODEL="Qwen/Qwen3-0.6B"
ARG QWEN_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
RUN python scripts/fetch_qwen.py --model "$QWEN_MODEL" --output /app/models/qwen-base ${QWEN_MODEL_REVISION:+--revision "$QWEN_MODEL_REVISION"}
ENV MAGICANVAS_QWEN_DIR=/app/models/qwen-base
ENV MAGICANVAS_QWEN_ENABLED=1

# A public adapter is fetched by default from the pinned champion repository.
# Evaluators can supply an alternate --build-arg QWEN_ADAPTER_REPO=... if swapping.
# The final submission stage below requires the verified pinned adapter.
ARG QWEN_ADAPTER_REPO="NeoliteNexus/magicanvas-qwen3-0.6b-lora"
ARG QWEN_ADAPTER_REVISION="09216ebcf0260889f84d21a723ce4ecacdc58e5b"
COPY scripts/fetch_qwen_lora.py ./scripts/fetch_qwen_lora.py
COPY scripts/verify_qwen_adapter.py ./scripts/verify_qwen_adapter.py
RUN if [ -n "$QWEN_ADAPTER_REPO" ]; then \
      if [ -n "$QWEN_ADAPTER_REVISION" ]; then \
        python scripts/fetch_qwen_lora.py --repo "$QWEN_ADAPTER_REPO" --revision "$QWEN_ADAPTER_REVISION"; \
      else \
        python scripts/fetch_qwen_lora.py --repo "$QWEN_ADAPTER_REPO"; \
      fi; \
    fi

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

# Final public submission path: no silent base-model-only fallback.
FROM runtime AS submission
ENV MAGICANVAS_REQUIRE_QWEN_ADAPTER=1
RUN test -n "$QWEN_ADAPTER_REPO" && test ${#QWEN_ADAPTER_REVISION} -eq 40 \
    && ! printf "%s" "$QWEN_ADAPTER_REVISION" | grep -q "[^0-9a-fA-F]" \
    && python scripts/verify_qwen_adapter.py --adapter /app/models/qwen3-layout-lora --model /app/models/qwen-base
