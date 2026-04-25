"""SchemaQuakeEnv — an OpenEnv v0.2.3 compliant Environment.

The environment wraps:
  - A freshly-loaded WorldState
  - A DriftScheduler (the Quake)
  - A RewardEngine (4-component reward)
  - A single SQState that tracks episode-level progress
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from .drift import DriftScheduler
from .prompts import SYSTEM_PROMPT, make_episode_brief
from .rewards import RewardBreakdown, RewardEngine
from .tools import TOOL_REGISTRY
from .types import (
    DriftType,
    EpisodeBrief,
    GroundTruth,
    SQAction,
    SQObservation,
    SQState,
    ToolName,
)
from .world import WorldState


class SchemaQuakeEnv(Environment[SQAction, SQObservation, SQState]):
    """Single-session SchemaQuake environment.

    Each call to `reset()` begins a new episode with:
      - a freshly loaded WorldState (flights, hotels, policies)
      - a new random user brief
      - a new DriftScheduler (may choose no drift 20% of the time)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        max_steps: int = 30,
        seed: Optional[int] = None,
        p_no_drift: float = 0.2,
    ) -> None:
        super().__init__(transform=None, rubric=None)
        self.max_steps = max_steps
        self._seed = seed
        self._p_no_drift = p_no_drift
        self._rng = random.Random(seed)

        # These are populated in reset().
        self._state: Optional[SQState] = None
        self._world: Optional[WorldState] = None
        self._scheduler: Optional[DriftScheduler] = None
        self._reward_engine: Optional[RewardEngine] = None

    # ------------------------------------------------------------------
    # OpenEnv required surface
    # ------------------------------------------------------------------
    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SchemaQuake",
            description=(
                "Travel-booking world where APIs, units, enums, and policies "
                "silently drift mid-episode. Agents are rewarded for noticing "
                "before they silently violate the user's intent."
            ),
            version="0.1.0",
            author="Ambuj Raj",
        )

    @property
    def state(self) -> SQState:
        assert self._state is not None, "Call reset() before state."
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SQObservation:
        if seed is not None:
            self._rng = random.Random(seed)

        brief = make_episode_brief(self._rng)
        scheduler = DriftScheduler.random(self._rng, p_no_drift=self._p_no_drift)
        world = WorldState.fresh()
        gt = GroundTruth(
            drift_type=scheduler.drift_type,
            drift_step=scheduler.drift_step,
        )
        state = SQState(
            episode_id=episode_id,
            step_count=0,
            brief=brief,
            ground_truth=gt,
            max_steps=kwargs.get("max_steps", self.max_steps),
        )

        self._state = state
        self._world = world
        self._scheduler = scheduler
        self._reward_engine = RewardEngine(brief=brief, ground_truth=gt)

        return SQObservation(
            message=(
                f"{SYSTEM_PROMPT}\n\nUser request:\n{brief.request_text}\n\n"
                "Begin by planning your first tool call."
            ),
            tool_result={},
            episode_brief=brief.model_dump(),
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: SQAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SQObservation:
        assert self._state is not None and self._world is not None
        assert self._scheduler is not None and self._reward_engine is not None

        state = self._state
        world = self._world
        state.step_count += 1
        current_step = state.step_count

        drift_fired_this_step = self._scheduler.maybe_fire(world, current_step)

        # Dispatch tool
        result, terminated = self._dispatch(action, world, state)
        state.action_log.append({
            "step": current_step,
            "tool": action.tool.value,
            "args": action.args,
            "confidence": action.confidence,
            "result": result,
        })

        # Per-step reward (all components except final task reward)
        self._reward_engine.on_step(
            tool=action.tool,
            tool_args=action.args,
            tool_result=result,
            confidence=action.confidence,
            current_step=current_step,
            drift_fired_this_step=drift_fired_this_step,
            world=world,
        )

        done = terminated or current_step >= state.max_steps
        state.terminated = done

        reward_breakdown: Optional[Dict[str, float]] = None
        step_reward: float = self._reward_engine.state.breakdown.total  # running total
        if done:
            # Compute final task reward based on submitted booking (if any).
            submitted = state.booking_id if action.tool == ToolName.SUBMIT else None
            final_bd: RewardBreakdown = self._reward_engine.on_terminal(
                submitted_booking_id=submitted, world=world
            )
            reward_breakdown = final_bd.as_dict()

        msg = self._summarize(action, result, drift_fired_this_step)

        return SQObservation(
            message=msg,
            tool_result=result,
            episode_brief=None,
            done=done,
            reward=self._reward_engine.state.breakdown.total,
            reward_breakdown=reward_breakdown,
            metadata={
                "step": current_step,
                "drift_fired_this_step": drift_fired_this_step,
                "drift_type": self._scheduler.drift_type.value,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _dispatch(
        self, action: SQAction, world: WorldState, state: SQState
    ) -> tuple[Dict[str, Any], bool]:
        """Run the chosen tool; return (result_dict, should_terminate)."""
        tool = action.tool
        args = action.args or {}

        if tool == ToolName.NOOP:
            return {"status": "noop"}, False

        if tool == ToolName.SUBMIT:
            booking_id = args.get("booking_id") or state.booking_id
            return (
                {
                    "status": "submitted",
                    "booking_id": booking_id,
                    "has_booking": booking_id is not None,
                },
                True,
            )

        if tool.value not in TOOL_REGISTRY:
            return {"error": "unknown_tool", "tool": tool.value}, False

        fn = TOOL_REGISTRY[tool.value]
        try:
            result = fn(world, **args)
        except TypeError as e:
            return {"error": "bad_args", "detail": str(e)}, False

        if tool == ToolName.BOOK and result.get("status") == "confirmed":
            state.booking_id = result["booking_id"]
            state.booked_item = {"item_id": result["item_id"], "kind": result["kind"]}

        if tool == ToolName.CANCEL and result.get("status") == "refunded":
            state.cancelled = True
            # A cancelled booking should no longer be the submission candidate.
            if state.booking_id == args.get("booking_id"):
                state.booking_id = None

        return result, False

    def _summarize(
        self, action: SQAction, result: Dict[str, Any], drift_fired: bool
    ) -> str:
        head = f"[step {self._state.step_count}] {action.tool.value}"
        if "error" in result:
            return f"{head} -> error: {result['error']}"
        if action.tool == ToolName.SUBMIT:
            return f"{head} -> episode ended."
        # Drift is intentionally NOT announced; the agent must notice.
        return f"{head} -> ok"
