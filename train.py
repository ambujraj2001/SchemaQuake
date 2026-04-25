"""Real GRPO training run for SchemaQuake.

Designed to be launched on a single GPU (L4 / A100). Reads config from env
vars so the same script works on Colab, HF Spaces, Modal, RunPod, etc.

Env vars (all optional):
  SQ_MODEL          model name (default: Qwen2.5-1.5B 4-bit)
  SQ_MAX_SEQ        max sequence length (default: 2048)
  SQ_NUM_PROMPTS    GRPO dataset size in prompts (default: 256)
  SQ_MAX_STEPS      max env steps per episode (default: 12)
  SQ_NUM_GEN        completions per prompt for GRPO group (default: 4)
  SQ_BATCH          per-device train batch size (default: 4)
  SQ_GRAD_ACCUM     gradient accumulation steps (default: 1)
  SQ_LR             learning rate (default: 1e-5)
  SQ_OUT            output dir (default: ./runs/schemaquake_grpo)

Usage:
  python train.py
"""
from __future__ import annotations

import json
import importlib
import logging
import os
import pkgutil
import re
import sys
import warnings

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
from datasets import Dataset
from unsloth import FastLanguageModel

def _resolve_grpo_symbols():
    import trl

    # 1) Common top-level export.
    if hasattr(trl, "GRPOConfig") and hasattr(trl, "GRPOTrainer"):
        return trl.GRPOConfig, trl.GRPOTrainer

    # 2) Known internal module layouts across TRL versions.
    candidates = [
        "trl.trainer.grpo_config",
        "trl.trainer.grpo_trainer",
        "trl.trainer.grpo",
        "trl.trainer",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "GRPOConfig") and hasattr(mod, "GRPOTrainer"):
            return mod.GRPOConfig, mod.GRPOTrainer

    # 3) Last resort: search TRL package modules for GRPO symbols.
    try:
        for m in pkgutil.walk_packages(trl.__path__, prefix="trl."):
            if "grpo" not in m.name.lower():
                continue
            try:
                mod = importlib.import_module(m.name)
            except Exception:
                continue
            if hasattr(mod, "GRPOConfig") and hasattr(mod, "GRPOTrainer"):
                return mod.GRPOConfig, mod.GRPOTrainer
    except Exception:
        pass

    raise ImportError(
        "GRPO is unavailable in the installed `trl` build. "
        "Could not find GRPOConfig/GRPOTrainer in top-level or trainer modules."
    )


GRPOConfig, GRPOTrainer = _resolve_grpo_symbols()

from schemaquake.env import SchemaQuakeEnv
from schemaquake.prompts import SYSTEM_PROMPT
from schemaquake.types import SQAction, ToolName

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)


MODEL_NAME = os.environ.get("SQ_MODEL", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
MAX_SEQ_LEN = int(os.environ.get("SQ_MAX_SEQ", 2048))
NUM_PROMPTS = int(os.environ.get("SQ_NUM_PROMPTS", 256))
MAX_STEPS_PER_EP = int(os.environ.get("SQ_MAX_STEPS", 12))
NUM_GEN = int(os.environ.get("SQ_NUM_GEN", 4))
BATCH = int(os.environ.get("SQ_BATCH", 4))
GRAD_ACCUM = int(os.environ.get("SQ_GRAD_ACCUM", 1))
LR = float(os.environ.get("SQ_LR", 1e-5))
OUT_DIR = os.environ.get("SQ_OUT", os.path.join(REPO_ROOT, "runs/schemaquake_grpo"))

if (BATCH * GRAD_ACCUM) % NUM_GEN != 0:
    raise SystemExit(
        f"Bad config: per_device_batch ({BATCH}) * grad_accum ({GRAD_ACCUM}) "
        f"must be a multiple of num_generations ({NUM_GEN})."
    )

print("=" * 60)
print(" SchemaQuake GRPO training")
print("=" * 60)
print(f" model           : {MODEL_NAME}")
print(f" num prompts     : {NUM_PROMPTS}")
print(f" episode steps   : {MAX_STEPS_PER_EP}")
print(f" num generations : {NUM_GEN}")
print(f" batch / accum   : {BATCH} / {GRAD_ACCUM}")
print(f" learning rate   : {LR}")
print(f" output dir      : {OUT_DIR}")
print("=" * 60, flush=True)


print("\nLoading model ...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
)
print("Model ready.", flush=True)


VALID_TOOLS = {t.value for t in ToolName}
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action(text: str) -> SQAction:
    m = JSON_RE.search(text or "")
    if not m:
        return SQAction(tool=ToolName.NOOP, confidence=0.0)
    try:
        d = json.loads(m.group(0))
        return SQAction(
            tool=ToolName(str(d.get("tool", "noop")).lower()),
            args=d.get("args") or {},
            confidence=d.get("confidence"),
        )
    except Exception:
        return SQAction(tool=ToolName.NOOP, confidence=0.0)


def build_prompt(history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def rollout_one(seed: int) -> float:
    env = SchemaQuakeEnv(max_steps=MAX_STEPS_PER_EP, p_no_drift=0.2)
    obs = env.reset(seed=seed, episode_id=f"tr-{seed}")
    history = [{"role": "user", "content": json.dumps(obs.episode_brief)}]
    while not obs.done:
        prompt = build_prompt(history)
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        text = tokenizer.decode(
            out[0][ids["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        obs = env.step(parse_action(text))
        history.append({"role": "assistant", "content": text})
        history.append({"role": "user", "content": json.dumps(obs.tool_result)[:1200]})
    return float((obs.reward_breakdown or {}).get("total", 0.0))


def shaped_reward(text: str, seed: int) -> float:
    """Format-shaping reward to break the zero-variance trap.

    Even before the model can solve the env, it gets variable signal for:
      - producing JSON
      - producing valid JSON with a known tool
      - producing a sane confidence
    On top of that we add half of the actual episode reward.
    """
    r = 0.0
    m = JSON_RE.search(text or "")
    if not m:
        return -0.2
    r += 0.1
    try:
        data = json.loads(m.group(0))
    except Exception:
        return r - 0.05
    r += 0.1
    if str(data.get("tool", "")).lower() in VALID_TOOLS:
        r += 0.2
    try:
        c = float(data.get("confidence"))
        if 0.0 <= c <= 1.0:
            r += 0.05
    except Exception:
        pass
    try:
        r += 0.5 * rollout_one(seed)
    except Exception:
        pass
    return r


def schemaquake_reward(completions, prompts=None, **_):
    out = []
    for comp, p in zip(completions, prompts):
        if isinstance(comp, str):
            text = comp
        elif isinstance(comp, list) and comp:
            text = comp[0].get("content", "") if isinstance(comp[0], dict) else str(comp[0])
        else:
            text = str(comp)
        seed = int(p.split(":")[-1]) if isinstance(p, str) else 0
        out.append(shaped_reward(text, seed))
    return out


train_ds = Dataset.from_list(
    [{"prompt": f"seed:{i}", "seed": i} for i in range(NUM_PROMPTS)]
)

bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

cfg = GRPOConfig(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_generations=NUM_GEN,
    max_prompt_length=512,
    max_completion_length=96,
    logging_steps=1,
    save_steps=50,
    report_to="none",
    bf16=bf16_supported,
    fp16=not bf16_supported,
    temperature=0.9,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=schemaquake_reward,
    args=cfg,
    train_dataset=train_ds,
    processing_class=tokenizer,
)

print("\nStarting training ...", flush=True)
trainer.train()
final_dir = os.path.join(OUT_DIR, "final")
trainer.save_model(final_dir)

with open(os.path.join(OUT_DIR, "log_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"\nDone. Adapter saved to {final_dir}", flush=True)
