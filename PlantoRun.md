## The big reveal: you've got three "fake AIs" built in

Your repo ships with **three agents**, and only one of them needs a real LLM:

| Agent | Needs a model? | What it does |
|---|---|---|
| **Random agent** | ❌ No | Picks tools at random. Baseline to prove training matters. |
| **Heuristic agent** | ❌ No | A hand-coded "smart" agent (your `HeuristicAgent` class). Proves the environment is solvable. |
| **LLM agent** | ✅ Yes (OpenAI API or local Ollama) | The real deal — plugs into an actual language model. |

For everything you want to do right now — testing, the demo, the eval plots — you only need the **first two**. No model, no GPU, no API key, nothing.

---

## Step 1 — One-time setup (5 minutes)

Open a terminal in your project folder:

```bash
cd "/Users/ambujraj/Documents/Personal Projects/hackathon-v2"

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,demo]"
```

Translation:
- `python3 -m venv .venv` → make a private Python sandbox in this folder (so you don't pollute your Mac).
- `source .venv/bin/activate` → step into the sandbox.
- `pip install -e ".[dev,demo]"` → install your project + the testing tools + the Gradio demo tools. The `-e` means "editable," so if you change code, it picks up changes without reinstalling.

**That's it. You never need to do this again unless you delete `.venv`.**

---

## Step 2 — Run the tests (30 seconds)

Your repo has 26 automated tests. This is the fastest sanity check that nothing is broken:

```bash
pytest -q
```

You should see something like:

```
..........................                                        [100%]
26 passed in 1.2s
```

That means: the fake world loads, the 7 tools work, the 4 drift operators fire correctly, the reward function math is right, and the environment follows the OpenEnv contract. **All without a single AI model involved.**

If you want to see *what* each test does:

```bash
pytest -v
```

---

## Step 3 — Run the random agent (30 seconds)

This is your "dumb baseline" — the AI equivalent of a monkey hitting buttons.

```bash
python -m eval.run_eval --agent random --episodes 50
```

This will:
1. Create 50 different booking tasks.
2. Let the random agent play each one (with drift firing silently).
3. Score each episode using your reward function.
4. Save a results JSON + a 4-panel plot to `eval_results/random_plot.png`.

You'll see a progress bar, and at the end, average reward ~**+0.62**. Open the PNG to see the plot.

```bash
open eval_results/random_plot.png
```

---

## Step 4 — Run the heuristic agent (30 seconds)

This is your **"this is possible"** proof. The hand-coded smart agent that actively looks for drift:

```bash
python -m eval.run_eval --agent heuristic --episodes 50
open eval_results/heuristic_plot.png
```

Expected average reward: **+0.90**. Silent-violation rate: **0%** (it never books the wrong thing).

**This is the single most important result in your project right now.** The gap between random (+0.62) and heuristic (+0.90) *proves* that the environment actually rewards drift-awareness. If a trained AI can match or beat the heuristic, that's your win condition.

---

## Step 5 — Run the Gradio demo (the thing you'll show the judges)

```bash
python demo/app.py
```

You'll see something like:

```
Running on local URL: http://127.0.0.1:7860
```

Open that URL in your browser. You'll get a web UI where you can:
- Type a booking request.
- Choose which drift operator fires (or none).
- Watch the agent play the episode **step by step** — every tool call, every probe, every decision.
- See the final reward breakdown.

The demo works **with the heuristic agent by default**, so no model is needed. This is perfect for:
- Practicing your pitch
- Recording your 2-minute demo video for YouTube
- Showing judges the live UI

---

## Step 6 — Want an actual LLM without training anything?

Two options, both free:

### Option A — Local Ollama (best for Mac)

Install Ollama (one-time):

```bash
brew install ollama
ollama pull qwen2.5:3b-instruct
ollama serve
```

Then in a different terminal:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama-anything
export OPENAI_MODEL=qwen2.5:3b-instruct

python -m eval.run_eval --agent llm --episodes 10
```

This runs the **untrained Qwen2.5-3B** on your Mac (it'll use the M-series GPU via Metal). Expected reward: probably somewhere between random and heuristic — because the raw model hasn't been trained on the drift skill yet. **This is exactly the "before training" number you want for your graph.**

### Option B — OpenAI API (if you already have a key)

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
python -m eval.run_eval --agent llm --episodes 10
```

GPT-4o-mini is good enough that it'll often match or beat the heuristic — which is a separate, cool story ("big frontier models already have this skill; ours proves a 3B model can learn it with just RL").

---

## Your five-command cheat sheet

Stick this next to your laptop during the hackathon:

```bash
source .venv/bin/activate

pytest -q                                                # 1. Are we healthy?
python -m eval.run_eval --agent random --episodes 50     # 2. The dumb baseline
python -m eval.run_eval --agent heuristic --episodes 50  # 3. The "it's possible" proof
python demo/app.py                                       # 4. The live UI (demo + video)

open eval_results/heuristic_plot.png                     # 5. The money shot for judges
```

Five commands. That's your whole local workflow, no GPU, no model weights, no cloud.

---

## What the judges actually see in all of this

- **The plots** (`eval_results/*.png`) → reward improvement graph. Headline number.
- **The Gradio demo** → live drama. The "wait, something feels off" moment on screen.
- **`src/schemaquake/drift.py`** → the 4 mischief functions, your novelty.
- **`src/schemaquake/rewards.py`** → the scoring logic, your thoughtfulness.
- **The 26 passing tests** → engineering maturity.

**All of this runs on your Mac in under 2 minutes, combined.** The GPU training part only matters for *one specific extra slide* (the "trained LLM beats the heuristic" graph). Everything else — the whole submission — is already alive on your laptop.

---

Want me to:
1. Actually run those commands now and show you the output, or
2. Walk through what the Gradio demo UI looks like and how to narrate it during your pitch?