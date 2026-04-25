"""Gradio demo app for SchemaQuake.

Runs a seeded episode with a selected agent and renders:
  - the user brief
  - the full action log with drift marker
  - per-component reward breakdown table
  - a JSON dump of the final booking

Deploy as a Hugging Face Space:
    huggingface-cli upload <space-id> demo/ --repo-type space
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from schemaquake import SchemaQuakeEnv


AGENT_REGISTRY = {
    "random (weak baseline)": RandomAgent,
    "heuristic (drift-aware)": HeuristicAgent,
}


def _format_action_log(log: List[Dict[str, Any]], drift_step: int | None) -> str:
    lines = []
    for entry in log:
        step = entry["step"]
        marker = "  🌋 DRIFT FIRES HERE" if drift_step is not None and step == drift_step else ""
        args = json.dumps(entry["args"], ensure_ascii=False)[:80]
        result_short = json.dumps(entry["result"], ensure_ascii=False)[:160]
        lines.append(
            f"step {step:>2d}  {entry['tool']:<16s}  args={args}{marker}\n"
            f"          -> {result_short}"
        )
    return "\n".join(lines)


def _format_reward_table(rb: Dict[str, float]) -> List[List[Any]]:
    order = ["task", "drift", "violation", "hedge", "step", "probe_spam", "total"]
    return [[k, f"{rb.get(k, 0.0):+.3f}"] for k in order]


def run(agent_choice: str, seed: int, p_no_drift: float) -> Tuple[str, str, List[List[Any]], str]:
    env = SchemaQuakeEnv(max_steps=30, p_no_drift=p_no_drift)
    agent = AGENT_REGISTRY[agent_choice]()

    obs = env.reset(seed=int(seed), episode_id=f"demo-{seed}")
    if hasattr(agent, "reset"):
        agent.reset()
    while not obs.done:
        obs = env.step(agent.act(obs))

    state = env.state
    gt = state.ground_truth
    brief_md = (
        f"### User request\n> {state.brief.request_text}\n\n"
        f"**Origin:** {state.brief.origin} | "
        f"**Destination:** {state.brief.destination} | "
        f"**Max price:** ₹{state.brief.max_price_rupees} | "
        f"**Must be refundable:** {state.brief.must_be_refundable}\n\n"
        f"### Ground truth (hidden from agent)\n"
        f"- Drift type: `{gt.drift_type.value}`\n"
        f"- Drift fires at step: `{gt.drift_step}`\n"
        f"- Drift detected at step: `{gt.drift_detected_at}`\n"
        f"- Silent violation: `{gt.silent_violation}`"
    )
    log_md = _format_action_log(state.action_log, gt.drift_step)
    table = _format_reward_table(obs.reward_breakdown or {})
    final_booking_md = (
        "```json\n"
        + json.dumps(
            env._world.bookings.get(state.booking_id) if state.booking_id else {},
            indent=2,
        )
        + "\n```"
    )
    return brief_md, f"```\n{log_md}\n```", table, final_booking_md


with gr.Blocks(title="SchemaQuake — live agent runner") as demo:
    gr.Markdown(
        "# SchemaQuake\n"
        "A travel-booking world where **APIs, units, enums, and policies silently drift**.\n"
        "Pick an agent and a seed, run one episode, and watch the reward breakdown."
    )
    with gr.Row():
        agent_choice = gr.Dropdown(
            choices=list(AGENT_REGISTRY), value="heuristic (drift-aware)",
            label="Agent",
        )
        seed = gr.Number(value=7, precision=0, label="Seed")
        p_no_drift = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="P(no drift)")
    run_btn = gr.Button("Run episode", variant="primary")
    brief_out = gr.Markdown()
    log_out = gr.Markdown()
    reward_table = gr.Dataframe(
        headers=["component", "value"], label="Reward breakdown"
    )
    booking_out = gr.Markdown(label="Final booking")
    run_btn.click(
        run, inputs=[agent_choice, seed, p_no_drift],
        outputs=[brief_out, log_out, reward_table, booking_out],
    )


if __name__ == "__main__":
    demo.launch()
