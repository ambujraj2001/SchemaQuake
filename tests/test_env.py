import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # access `agents/`

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from schemaquake import SchemaQuakeEnv, SQAction, ToolName
from schemaquake.types import DriftType


def _run(env, agent, seed):
    obs = env.reset(seed=seed, episode_id=f"ep-{seed}")
    agent.reset() if hasattr(agent, "reset") else None
    while not obs.done:
        obs = env.step(agent.act(obs))
    return obs, env.state


def test_env_reset_returns_brief():
    env = SchemaQuakeEnv(seed=0, p_no_drift=1.0)
    obs = env.reset(seed=1)
    assert obs.episode_brief is not None
    assert obs.done is False


def test_env_full_episode_with_heuristic():
    env = SchemaQuakeEnv(seed=0, p_no_drift=1.0)  # no drift for this one
    obs, state = _run(env, HeuristicAgent(), seed=1)
    assert obs.done is True
    assert obs.reward_breakdown is not None
    assert state.ground_truth.drift_type == DriftType.NONE


def test_env_drift_occurs():
    env = SchemaQuakeEnv(seed=0, p_no_drift=0.0, max_steps=30)
    obs, state = _run(env, HeuristicAgent(), seed=2)
    assert state.ground_truth.drift_type != DriftType.NONE


def test_env_heuristic_outperforms_random_on_average():
    def avg(agent_cls, n=8):
        env = SchemaQuakeEnv(seed=0, p_no_drift=0.2, max_steps=30)
        rewards = []
        for i in range(n):
            obs, _ = _run(env, agent_cls(), seed=100 + i)
            rewards.append(obs.reward_breakdown["total"])
        return sum(rewards) / len(rewards)

    r_random = avg(RandomAgent)
    r_heur = avg(HeuristicAgent)
    # The heuristic probes/re-reads policies and filters by canonical price.
    # It should beat the random baseline by a clear margin.
    assert r_heur > r_random, f"heuristic {r_heur:.3f} <= random {r_random:.3f}"


def test_env_submit_without_booking_yields_zero_task_reward():
    env = SchemaQuakeEnv(seed=0, p_no_drift=1.0)
    env.reset(seed=42)
    obs = env.step(SQAction(tool=ToolName.SUBMIT, args={}, confidence=0.5))
    assert obs.done is True
    assert obs.reward_breakdown["task"] == 0.0
