"""Console entry points defined in pyproject.toml."""
from __future__ import annotations


def eval_main() -> None:  # pragma: no cover - thin wrapper
    from eval.run_eval import main
    main()


def play_main() -> None:  # pragma: no cover
    """Interactive one-episode runner, useful for sanity checks."""
    import json

    from .env import SchemaQuakeEnv
    from .types import SQAction, ToolName

    env = SchemaQuakeEnv()
    obs = env.reset(seed=0)
    print(obs.message)
    while not obs.done:
        raw = input("action JSON > ").strip()
        if not raw:
            continue
        data = json.loads(raw)
        action = SQAction(
            tool=ToolName(data.get("tool", "noop")),
            args=data.get("args") or {},
            confidence=data.get("confidence"),
        )
        obs = env.step(action)
        print(json.dumps(obs.model_dump(), indent=2, default=str))
