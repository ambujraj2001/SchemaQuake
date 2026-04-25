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

ARG REPO_URL=https://github.com/ambujraj2001/SchemaQuake.git
ARG REPO_REF=main
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app

RUN pip install --no-cache-dir \
    "unsloth" \
    "trl>=0.11" "transformers>=4.44" "peft>=0.12" \
    "accelerate>=0.33" "datasets>=2.20" "bitsandbytes>=0.43" \
    "openenv-core>=0.2.3" \
    "huggingface_hub>=0.24" "gradio>=4.40"

RUN pip install --no-cache-dir -e ".[dev,agents]"

EXPOSE 7860

CMD ["python", "space/app.py"]
