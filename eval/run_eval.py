"""Evaluate an agent over N seeded episodes and produce a reward breakdown plot.

Usage:
    python -m eval.run_eval --agent heuristic --episodes 50 --seed 0
    python -m eval.run_eval --agent random    --episodes 50 --seed 0
    python -m eval.run_eval --agent llm       --episodes 20 --seed 0  (requires OPENAI_API_KEY)

Produces:
    eval_results/<agent>_episodes.json   — full per-episode breakdowns
    eval_results/<agent>_summary.json    — aggregate metrics
    eval_results/<agent>_plot.png        — 4-panel comparison plot
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from schemaquake import SchemaQuakeEnv, SQAction, ToolName


AGENTS = {
    "random": "agents.random_agent:RandomAgent",
    "heuristic": "agents.heuristic_agent:HeuristicAgent",
    "llm": "agents.llm_agent:LLMAgent",
}


def _load_agent(name: str):
    spec = AGENTS[name]
    mod_path, cls_name = spec.split(":")
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


@dataclass
class EpisodeResult:
    episode_id: int
    drift_type: str
    drift_step: int | None
    drift_detected_at: int | None
    silent_violation: bool
    steps: int
    booking_submitted: bool
    reward_breakdown: Dict[str, float]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def run_episode(env: SchemaQuakeEnv, agent, seed: int) -> EpisodeResult:
    obs = env.reset(seed=seed, episode_id=f"ep-{seed}")
    if hasattr(agent, "reset"):
        agent.reset()

    last_obs = obs
    while not last_obs.done:
        action = agent.act(last_obs)
        last_obs = env.step(action)

    state = env.state
    gt = state.ground_truth
    return EpisodeResult(
        episode_id=seed,
        drift_type=gt.drift_type.value,
        drift_step=gt.drift_step,
        drift_detected_at=gt.drift_detected_at,
        silent_violation=gt.silent_violation,
        steps=state.step_count,
        booking_submitted=bool(state.booking_id),
        reward_breakdown=last_obs.reward_breakdown or {},
    )


def aggregate(results: List[EpisodeResult]) -> Dict[str, float]:
    if not results:
        return {}
    rb = [r.reward_breakdown for r in results]
    keys = ["task", "drift", "violation", "hedge", "step", "probe_spam", "total"]
    agg = {k: sum(rbi.get(k, 0.0) for rbi in rb) / len(rb) for k in keys}
    agg["silent_violation_rate"] = sum(r.silent_violation for r in results) / len(results)
    drift_eps = [r for r in results if r.drift_type != "none"]
    agg["drift_detection_rate"] = (
        sum(1 for r in drift_eps if r.drift_detected_at is not None) / max(1, len(drift_eps))
    )
    agg["task_success_rate"] = sum(1 for r in rb if r.get("task", 0.0) >= 1.0) / len(rb)
    agg["avg_steps"] = sum(r.steps for r in results) / len(results)
    return agg


def plot_breakdown(results: List[EpisodeResult], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rb = [r.reward_breakdown for r in results]
    tasks = [r.get("task", 0.0) for r in rb]
    drifts = [r.get("drift", 0.0) for r in rb]
    viols = [r.get("violation", 0.0) for r in rb]
    hedges = [r.get("hedge", 0.0) for r in rb]
    totals = [r.get("total", 0.0) for r in rb]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, weight="bold")

    axes[0, 0].plot(totals, marker="o", linewidth=1, color="tab:blue")
    axes[0, 0].axhline(np.mean(totals), linestyle="--", color="black", alpha=0.5,
                       label=f"mean {np.mean(totals):.2f}")
    axes[0, 0].set_title("Total reward per episode")
    axes[0, 0].set_xlabel("episode"); axes[0, 0].set_ylabel("total reward")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].bar(range(len(tasks)), tasks, color="tab:green", alpha=0.7)
    axes[0, 1].set_title(f"Task reward (success rate "
                         f"{sum(t>=1 for t in tasks)/max(1,len(tasks)):.0%})")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].bar(range(len(drifts)), drifts, color="tab:orange", alpha=0.7)
    axes[1, 0].set_title("Drift-detection reward")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].bar(range(len(viols)), viols, color="tab:red", alpha=0.7)
    axes[1, 1].set_title(
        f"Silent-violation penalty "
        f"({sum(v < 0 for v in viols)} violations / {len(viols)} episodes)"
    )
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=list(AGENTS), default="heuristic")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--p-no-drift", type=float, default=0.2)
    args = parser.parse_args()

    env = SchemaQuakeEnv(max_steps=args.max_steps, p_no_drift=args.p_no_drift)
    AgentCls = _load_agent(args.agent)
    agent = AgentCls()

    results: List[EpisodeResult] = []
    for i in range(args.episodes):
        seed = args.seed + i
        r = run_episode(env, agent, seed=seed)
        results.append(r)
        print(
            f"ep {i+1:>3d}/{args.episodes} drift={r.drift_type:<14s} "
            f"task={r.reward_breakdown.get('task', 0.0):+.2f} "
            f"total={r.reward_breakdown.get('total', 0.0):+.2f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.agent}_episodes.json").write_text(
        json.dumps([r.to_json() for r in results], indent=2)
    )
    agg = aggregate(results)
    (out_dir / f"{args.agent}_summary.json").write_text(json.dumps(agg, indent=2))
    plot_breakdown(
        results, out_dir / f"{args.agent}_plot.png",
        title=f"SchemaQuake — {args.agent} agent ({args.episodes} episodes)",
    )

    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k:<22s} = {v:.3f}")
    print(f"\nArtifacts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
