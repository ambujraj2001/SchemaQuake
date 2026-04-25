"""Reward function for SchemaQuake.

The reward is composed of four interpretable components. Each component is
logged separately so training curves can show *which* skill is improving.

Components:
  R_task        : +1.0 if final booking satisfies the user brief, else 0
                  (partial credit for valid-but-suboptimal bookings)
  R_drift       : +0.3 for detecting drift within DRIFT_DETECT_WINDOW steps,
                  scaled by recency (earlier = more reward)
  R_violation   : -1.0 for booking in silent violation of the (post-drift)
                  policy (refundability or max-price)
  R_hedge       : +0.1 for a well-calibrated ask_user; -0.1 for a wasteful
                  ask_user when confidence was already high

Plus small shaping terms:
  R_step        : -0.01 per step (encourage efficiency)
  R_probe_spam  : -0.02 per probe_schema beyond the 3rd (avoid degenerate
                  always-probe policies)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import DriftType, EpisodeBrief, GroundTruth, ToolName
from .world import WorldState


DRIFT_DETECT_WINDOW = 4           # steps after drift to still count as "detected"
HIGH_CONFIDENCE_THRESHOLD = 0.85  # above this, asking the user is wasteful
LOW_CONFIDENCE_THRESHOLD = 0.55   # below this, asking the user is smart
MAX_HEDGE_BONUS = 0.2             # per-episode cap: prevents ask_user spam
MAX_HEDGE_PENALTY = -0.5          # per-episode cap: bounded punishment


@dataclass
class RewardBreakdown:
    task: float = 0.0
    drift: float = 0.0
    violation: float = 0.0
    hedge: float = 0.0
    step: float = 0.0
    probe_spam: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.task + self.drift + self.violation
            + self.hedge + self.step + self.probe_spam
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "task": self.task,
            "drift": self.drift,
            "violation": self.violation,
            "hedge": self.hedge,
            "step": self.step,
            "probe_spam": self.probe_spam,
            "total": self.total,
        }


@dataclass
class RewardState:
    """Per-episode running reward computed incrementally by `RewardEngine`."""
    breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)
    probe_count: int = 0

    def as_dict(self) -> Dict[str, float]:
        return self.breakdown.as_dict()


class RewardEngine:
    """Incremental reward computation across an episode.

    Call `on_step` after every action with rich context. Call `on_terminal`
    once when the episode ends to compute final task reward.
    """

    def __init__(self, brief: EpisodeBrief, ground_truth: GroundTruth) -> None:
        self.brief = brief
        self.gt = ground_truth
        self.state = RewardState()

    # ----- per-step incremental reward -----
    def on_step(
        self,
        tool: ToolName,
        tool_args: Dict[str, Any],
        tool_result: Dict[str, Any],
        confidence: Optional[float],
        current_step: int,
        drift_fired_this_step: bool,
        world: WorldState,
    ) -> float:
        delta = 0.0
        self.state.breakdown.step += -0.01
        delta += -0.01

        if tool == ToolName.PROBE_SCHEMA:
            self.state.probe_count += 1
            if self.state.probe_count > 3:
                self.state.breakdown.probe_spam += -0.02
                delta += -0.02

        # Drift-detection reward: did the agent probe or re-read policies
        # within DRIFT_DETECT_WINDOW steps of drift firing?
        if self.gt.drift_type != DriftType.NONE and self.gt.drift_step is not None:
            if tool in (ToolName.PROBE_SCHEMA, ToolName.READ_POLICIES):
                if (
                    self.gt.drift_detected_at is None
                    and current_step >= self.gt.drift_step
                    and current_step - self.gt.drift_step <= DRIFT_DETECT_WINDOW
                ):
                    # Linear decay: +0.3 same step, -0.05 per subsequent step.
                    lag = current_step - self.gt.drift_step
                    bonus = max(0.1, 0.3 - 0.05 * lag)
                    self.state.breakdown.drift += bonus
                    delta += bonus
                    self.gt.drift_detected_at = current_step

        # Uncertainty-gated ask_user shaping. Clamped per-episode so that
        # an agent can't accumulate unlimited reward by repeatedly asking.
        if tool == ToolName.ASK_USER and confidence is not None:
            if confidence <= LOW_CONFIDENCE_THRESHOLD:
                if self.state.breakdown.hedge < MAX_HEDGE_BONUS:
                    bonus = min(0.1, MAX_HEDGE_BONUS - self.state.breakdown.hedge)
                    self.state.breakdown.hedge += bonus
                    delta += bonus
                self.gt.asked_user_when_uncertain += 1
            elif confidence >= HIGH_CONFIDENCE_THRESHOLD:
                if self.state.breakdown.hedge > MAX_HEDGE_PENALTY:
                    pen = max(-0.1, MAX_HEDGE_PENALTY - self.state.breakdown.hedge)
                    self.state.breakdown.hedge += pen
                    delta += pen
                self.gt.asked_user_when_confident += 1

        # Silent-violation detection happens on book() itself: if the booking
        # violates the user brief evaluated against canonical (drift-aware)
        # WorldState, flag it here. This is the punishing signal.
        if tool == ToolName.BOOK and tool_result.get("status") == "confirmed":
            violation = self._violation_check(tool_result, world)
            if violation:
                self.state.breakdown.violation += -1.0
                delta += -1.0
                self.gt.silent_violation = True

        return delta

    def _violation_check(
        self, booking_result: Dict[str, Any], world: WorldState
    ) -> bool:
        price = booking_result.get("price_charged", 10**9)
        refundable = bool(booking_result.get("refundable", False))
        if price > self.brief.max_price_rupees:
            return True
        if self.brief.must_be_refundable and not refundable:
            return True
        return False

    # ----- terminal reward -----
    def on_terminal(
        self,
        submitted_booking_id: Optional[str],
        world: WorldState,
    ) -> RewardBreakdown:
        if submitted_booking_id is None:
            return self.state.breakdown  # no task reward
        rec = world.bookings.get(submitted_booking_id)
        if rec is None:
            return self.state.breakdown

        # Canonical, drift-aware evaluation of the final booking.
        canonical_price = rec["canonical_price_rupees"]
        refundable = rec["refundable"]

        price_ok = canonical_price <= self.brief.max_price_rupees
        refund_ok = (not self.brief.must_be_refundable) or refundable

        if price_ok and refund_ok:
            # Full task reward, with a small bonus for undercutting max price.
            slack = (self.brief.max_price_rupees - canonical_price) / max(
                1, self.brief.max_price_rupees
            )
            self.state.breakdown.task += 1.0 + 0.2 * slack
        elif price_ok or refund_ok:
            self.state.breakdown.task += 0.3  # partial credit
        # else: zero task reward

        return self.state.breakdown
