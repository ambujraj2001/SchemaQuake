---
title: SchemaQuake GRPO Training
emoji: "🌊"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# SchemaQuake GRPO Training Space

Runs `train.py` once on the Space's GPU, saves the LoRA adapter under
`runs/schemaquake_grpo/final`, and uploads it to the Hugging Face Hub
if `HF_TOKEN` and `HF_REPO_ID` are configured as Space secrets.

## How it's deployed

The `Dockerfile` at the **repository root** is what HF builds. It
`COPY`s the whole repo into the image, installs deps, and runs
`space/app.py` as the entrypoint. No GitHub round-trip on rebuild.

## One-time setup

1. Create a private Space on Hugging Face:
   - SDK: **Docker**
   - Hardware: **1x Nvidia L4** (`$0.80/hr`) recommended
2. Settings → Variables and secrets:
   - Secret `HF_TOKEN`     — HF write token
   - Secret `HF_REPO_ID`   — e.g. `ambujraj2001/schemaquake-grpo-lora`

## Pushing code directly (no GitHub round-trip)

From your local clone of this repo:

```bash
# one-time
git remote add space https://huggingface.co/spaces/<user>/<space-name>

# every deploy
git push space main:main --force
```

When prompted: **username = your HF username**, **password = an HF
write token** (Settings → Access Tokens on huggingface.co).

The Space rebuilds automatically on each push.

## Tuning

Add any of these as Space *variables* (not secrets):

```
SQ_MODEL=unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
SQ_NUM_PROMPTS=256
SQ_NUM_GEN=4
SQ_BATCH=4
SQ_GRAD_ACCUM=1
SQ_LR=1e-5
SQ_MAX_STEPS=12
```
