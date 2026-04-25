# SchemaQuake

> **Every LLM-agent demo cheats by freezing the world. Real APIs rename their fields on a Tuesday. Real T&Cs get a footnote no one reads.**
> SchemaQuake is an OpenEnv environment where the ground silently moves under the agent — and a reward function that teaches it to notice.

[![OpenEnv 0.2.3](https://img.shields.io/badge/OpenEnv-0.2.3-blue)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)]()
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)]()

Built for the **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology — Grand Finale (Apr 25–26, 2026)**.
Theme: **World Modeling (3.1 Professional Tasks)** — targets the Patronus AI sub-theme on schema drift and doubles on Scaler AI Labs' enterprise-workflow theme.

---

## The one-sentence idea

> **SchemaQuake is a training ground where the rules of the game silently change mid-game, and the agent learns to notice before it screws up.**

## The hook, in 30 seconds

A travel-booking agent is told: _"Book a refundable flight from BLR to DEL for under ₹8,000."_ On step 5, the API silently renames `price` to `amount_inr`, or changes units from rupees to paise, or mutates `refundable: true/false` into `refund_tier: "full"|"partial"|"none"`, or quietly updates the cancellation-policy document. The agent is never told. A careless agent books a non-refundable flight for ₹540,000 and confidently reports success. A well-trained agent probes the schema, re-reads policies, asks the user when uncertain, and still closes the booking.

**That reactive "wait, something feels off" skill is what SchemaQuake teaches.**

---

## Environment at a glance

- **Action space:** `search_flights`, `search_hotels`, `read_policies`, `probe_schema`, `book`, `cancel`, `ask_user`, `submit`.
- **Observation space:** last tool result + episode brief on reset. Drift is **never** announced.
- **State (hidden):** user brief, drift schedule, bookings, silent-violation flag, drift-detection latency.
- **Drift operators:** `rename_field`, `change_units`, `mutate_enum`, `update_policy`. Exactly one fires per episode (with 20% no-drift control).
- **Max steps:** 30 per episode.

## Reward model

Four interpretable components so training curves can prove _which_ skill is improving.

| Component     | Sign | Fires when                                                                                                      |
| ------------- | ---- | --------------------------------------------------------------------------------------------------------------- |
| `task`        | +    | Final booking matches the user brief (price & refundability). Partial credit for near-misses.                   |
| `drift`       | +    | Agent calls `probe_schema` or `read_policies` within 4 steps of drift firing. Decays with lag.                  |
| `violation`   | −    | Agent silently books in violation of the (post-drift) brief — e.g. non-refundable when user asked refundable.   |
| `hedge`       | ±    | `ask_user` with low confidence → +0.1. With high confidence → −0.1. This trains _calibrated_ hedging.           |
| `step`        | −    | 0.01 per step: light efficiency pressure.                                                                       |
| `probe_spam`  | −    | 0.02 per `probe_schema` beyond the third: prevents degenerate "always probe" policies.                          |

## Baseline numbers (50 episodes, seed 0, mixed drift)

| Agent      | Task success | Silent-violation rate | Total reward |
| ---------- | ------------ | --------------------- | ------------ |
| Random     | 72%          | **20%**               | +0.62        |
| Heuristic  | 86%          | **0%**                | +0.90        |

The 20 → 0 percentage-point drop in silent violations is the demo you show the judges.

---

## Repository layout

```
src/schemaquake/       OpenEnv-compliant environment package
 ├─ types.py           Pydantic SQAction / SQObservation / SQState
 ├─ world.py           WorldState with canonical price/refund helpers
 ├─ tools.py           Tool implementations over WorldState
 ├─ drift.py           4 drift operators + DriftScheduler
 ├─ rewards.py         RewardEngine with 4 components
 ├─ prompts.py         Episode brief generator + system prompt
 ├─ env.py             SchemaQuakeEnv(Environment)
 └─ data/              flights.json, hotels.json, policies.md
agents/                RandomAgent, HeuristicAgent, LLMAgent
eval/run_eval.py       N-episode harness → JSON + 4-panel plot
tests/                 26 unit & integration tests (pytest)
notebooks/train_grpo.ipynb   Unsloth + TRL GRPO training
demo/app.py            Gradio HF Space UI
docs/                  BLOG.md, PITCH.md
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,demo]"
pytest -q                                    # 26 tests, all should pass
python -m eval.run_eval --agent heuristic --episodes 50
open eval_results/heuristic_plot.png
python demo/app.py                           # Gradio demo
```

## Environment variables

**None are required** for the core submission — the environment, tests,
Random/Heuristic agents, eval, and Gradio demo run fully offline.

Optional env vars (see `.env.example` for the full list):

- `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL` — only needed for
  `python -m eval.run_eval --agent llm`.
- `HUGGINGFACE_HUB_TOKEN` — only for pushing the HF Space / model checkpoint.
- `WANDB_API_KEY` — only for live training curves during GRPO.

```bash
cp .env.example .env                         # then fill in real values
set -a && source .env && set +a              # load into the shell
```


## Training

See `notebooks/train_grpo.ipynb` for a self-contained Colab-ready Unsloth + TRL GRPO pipeline on Qwen2.5-3B-Instruct (LoRA rank 16, 4-bit). Recommended run on a single A100 40 GB.

## OpenEnv compliance

- `SchemaQuakeEnv` inherits `openenv.core.env_server.interfaces.Environment[SQAction, SQObservation, SQState]`.
- Actions, observations, and state are Pydantic models inheriting the OpenEnv base classes (`extra="forbid"`, `validate_assignment=True`).
- `reset()`, `step()`, and the `state` property implement the abstract contract.
- `get_metadata()` returns an `EnvironmentMetadata` object.
- `SUPPORTS_CONCURRENT_SESSIONS = True` — no shared mutable state.

## Citation

If you use SchemaQuake in training or research:

```
@misc{schemaquake2026,
  author = {Ambuj Raj},
  title  = {SchemaQuake: A Drift-Native OpenEnv Environment for LLM Agents},
  year   = {2026},
  url    = {https://github.com/<your-handle>/schemaquake}
}
```

## License

Apache-2.0.
