FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_home \
    TRANSFORMERS_VERBOSITY=error \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    MKL_THREADING_LAYER=GNU \
    MKL_SERVICE_FORCE_INTEL=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Pin torch + torchvision together so unsloth can't break ABI
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall \
        torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu121

# 2. Install heavy ML deps before COPY so source edits don't bust the cache.
RUN pip install --no-cache-dir \
        "transformers>=4.44,<4.50" \
        "peft>=0.12,<0.14" \
        "accelerate>=0.33,<0.35" \
        "trl==0.12.2" \
        "datasets>=2.20" \
        "bitsandbytes>=0.43" \
        "openenv-core>=0.2.3" \
        "huggingface_hub>=0.24" \
        "gradio>=4.40" \
        "xformers==0.0.27.post2" \
        --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --no-deps "unsloth==2024.10.7" "unsloth-zoo==2024.11.0" && \
    pip install --no-cache-dir --force-reinstall --no-deps \
        "trl==0.12.2" "transformers==4.46.3" "tokenizers==0.20.3"

# 3. Re-pin torch/torchvision in case any transitive dep tried to upgrade them.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 \
        --index-url https://download.pytorch.org/whl/cu121

# 4. Project source goes last so code-only edits use the cached pip layers.
COPY . /app

RUN pip install --no-cache-dir -e ".[dev,agents]"

# 5. Validate core import/runtime compatibility at build time.
RUN python space/preflight.py --build-check

EXPOSE 7860

CMD ["python", "space/app.py"]
