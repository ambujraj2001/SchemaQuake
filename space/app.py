"""HF Space entry point.

Boots a tiny Gradio status page, kicks off `train.py` in a background
thread, streams logs to the UI, and uploads the trained LoRA adapter to
the user's Hugging Face Hub when training finishes.

Required Space secrets:
  HF_TOKEN     write-access token for upload (Settings -> Variables)

Optional Space variables (Settings -> Variables):
  HF_REPO_ID         target HF model repo (default: <user>/schemaquake-grpo-lora)
  SQ_MODEL           base model (default: Qwen2.5-1.5B-Instruct-bnb-4bit)
  SQ_NUM_PROMPTS     dataset size  (default: 256)
  SQ_NUM_GEN         GRPO group   (default: 4)
  SQ_BATCH           batch size   (default: 4)
  SQ_GRAD_ACCUM      grad accum   (default: 1)
  SQ_LR              learning rate (default: 1e-5)
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import threading
import time

import gradio as gr

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
LOG_FILE = "/tmp/sq_training.log"
DONE_FILE = "/tmp/sq_training.done"
ADAPTER_DIR = os.path.join(REPO_ROOT, "runs", "schemaquake_grpo", "final")

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")  # e.g. ambujraj2001/schemaquake-grpo


def _now() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _append(msg: str) -> None:
    line = f"[{_now()}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    sys.stdout.write(line)
    sys.stdout.flush()


def upload_adapter() -> None:
    if not (HF_TOKEN and HF_REPO_ID):
        _append("HF_TOKEN or HF_REPO_ID missing; skipping upload.")
        return
    if not os.path.exists(ADAPTER_DIR):
        _append(f"adapter dir not found: {ADAPTER_DIR}")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_REPO_ID, exist_ok=True, private=True)
        _append(f"uploading {ADAPTER_DIR} -> {HF_REPO_ID}")
        api.upload_folder(
            folder_path=ADAPTER_DIR,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="GRPO trained adapter from SchemaQuake Space",
        )
        _append(f"upload complete: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        _append(f"upload failed: {e!r}")


def run_training() -> None:
    _append("running preflight checks")
    preflight = subprocess.run(
        [sys.executable, "-u", "space/preflight.py"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if preflight.stdout:
        with open(LOG_FILE, "a") as f:
            f.write(preflight.stdout)
        sys.stdout.write(preflight.stdout)
        sys.stdout.flush()
    if preflight.returncode != 0:
        _append("preflight failed; not starting train.py")
        open(DONE_FILE, "w").write(str(preflight.returncode))
        return

    _append("starting train.py")
    cmd = [sys.executable, "-u", "train.py"]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        with open(LOG_FILE, "a") as f:
            f.write(line)
        sys.stdout.write(line)
    proc.wait()
    _append(f"train.py exited with code {proc.returncode}")
    if proc.returncode == 0:
        upload_adapter()
    open(DONE_FILE, "w").write(str(proc.returncode))


def boot() -> None:
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)
    _append("Space boot")
    _append(f"REPO_ROOT={REPO_ROOT}")
    _append(f"HF_REPO_ID={HF_REPO_ID or '(unset; will not upload)'}")
    _append(f"HAS_HF_TOKEN={bool(HF_TOKEN)}")
    threading.Thread(target=run_training, daemon=True).start()


def read_logs() -> str:
    if not os.path.exists(LOG_FILE):
        return "(waiting for training to start...)"
    with open(LOG_FILE) as f:
        data = f.read()
    return data[-20000:]


def status_text() -> str:
    if os.path.exists(DONE_FILE):
        rc = open(DONE_FILE).read().strip()
        return f"DONE (exit {rc})"
    return "RUNNING"


with gr.Blocks(title="SchemaQuake GRPO") as demo:
    gr.Markdown("# SchemaQuake GRPO Training\nThis Space runs `train.py` once, saves the LoRA adapter, and (if `HF_TOKEN` and `HF_REPO_ID` are set) uploads it to the Hugging Face Hub.")

    status_box = gr.Textbox(label="status", value="(starting...)", interactive=False)
    log_box = gr.Textbox(label="logs (tail 20K chars)", value="", lines=30, interactive=False)
    refresh = gr.Button("refresh")

    def _refresh():
        return status_text(), read_logs()

    refresh.click(_refresh, outputs=[status_box, log_box])
    demo.load(_refresh, outputs=[status_box, log_box])

if __name__ == "__main__":
    boot()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
