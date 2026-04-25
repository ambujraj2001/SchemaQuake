"""Typed Action / Observation / State for SchemaQuake.

These are Pydantic models that inherit from OpenEnv base classes so that
`SchemaQuakeEnv` is a fully compliant `openenv.core` Environment.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State


class ToolName(str, Enum):
    SEARCH_FLIGHTS = "search_flights"
    SEARCH_HOTELS = "search_hotels"
    READ_POLICIES = "read_policies"
    PROBE_SCHEMA = "probe_schema"
    BOOK = "book"
    CANCEL = "cancel"
    ASK_USER = "ask_user"
    SUBMIT = "submit"
    NOOP = "noop"


class SQAction(Action):
    """A SchemaQuake action is a single tool call.

    `tool` selects which tool to invoke; `args` is the payload.
    `confidence` is an optional self-reported probability in [0, 1] that the
    agent will use when calling `ask_user`. The reward function treats
    low-confidence + ask_user as correct hedging behaviour.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tool: ToolName = Field(description="Which tool to invoke.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments.")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Optional self-reported confidence used for ask_user gating.",
    )


class SQObservation(Observation):
    """Observation returned after every step.

    `tool_result` is the structured response of the invoked tool.
    `message` is a human-readable summary for LLM consumption.
    `reward_breakdown` is populated only on terminal steps; it is NEVER visible
    to the agent (policy) but is used for logging and post-hoc analysis.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    message: str = Field(description="Human-readable step summary for the agent.")
    tool_result: Dict[str, Any] = Field(
        default_factory=dict, description="Structured response of the last tool call."
    )
    episode_brief: Optional[Dict[str, Any]] = Field(
        default=None, description="User brief, only set on reset()."
    )
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None, description="Per-component reward, set on terminal step only."
    )


class DriftType(str, Enum):
    NONE = "none"
    RENAME_FIELD = "rename_field"
    CHANGE_UNITS = "change_units"
    MUTATE_ENUM = "mutate_enum"
    UPDATE_POLICY = "update_policy"


class EpisodeBrief(State):
    """User-facing brief. Copied verbatim into the first observation."""
    request_text: str
    origin: str
    destination: str
    max_price_rupees: int
    must_be_refundable: bool


class GroundTruth(State):
    """Hidden ground truth for reward computation. Never shown to the agent."""
    drift_type: DriftType = DriftType.NONE
    drift_step: Optional[int] = None        # step index at which drift fires
    drift_detected_at: Optional[int] = None # step index at which agent probed/re-read
    silent_violation: bool = False          # agent booked in violation post-drift
    asked_user_when_uncertain: int = 0      # bonus counter
    asked_user_when_confident: int = 0      # penalty counter


class SQState(State):
    """Full environment state (server-side only, not all exposed to the agent)."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    brief: Optional[EpisodeBrief] = None
    ground_truth: Optional[GroundTruth] = None
    max_steps: int = 30
    terminated: bool = False
    booking_id: Optional[str] = None
    booked_item: Optional[Dict[str, Any]] = None
    cancelled: bool = False
    action_log: List[Dict[str, Any]] = Field(default_factory=list)
