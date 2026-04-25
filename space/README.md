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

## Setup

1. Create a new Space:
   - SDK: **Docker**
   - Visibility: **Private**
2. Push **only** the contents of `space/` to the Space repo:
   - `Dockerfile`
   - `app.py`
   - `README.md` (this file)
3. Settings -> Hardware: pick **1x Nvidia L4** ($0.80/hr) or **A100 large** ($2.50/hr).
4. Settings -> Variables and secrets:
   - `HF_TOKEN`     - HF write token
   - `HF_REPO_ID`   - target model repo, e.g. `ambujraj2001/schemaquake-grpo-lora`
5. Open the Space - training starts automatically.

## Tuning

Add any of these as Space *variables* (not secrets):

```
SQ_MODEL=unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
SQ_NUM_PROMPTS=256
SQ_NUM_GEN=4
SQ_BATCH=4
SQ_GRAD_ACCUM=1
SQ_LR=1e-5
```
