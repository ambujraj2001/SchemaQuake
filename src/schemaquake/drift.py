"""Drift operators — the heart of SchemaQuake.

Each operator mutates `WorldState` in place to change the *visible* schema or
policy in a way the agent is never told about. The *semantics* remain intact:
the environment can still compute canonical prices and refundability for
reward purposes, through the helpers on WorldState.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .types import DriftType
from .world import WorldState


def apply_rename_field(world: WorldState) -> None:
    """Rename `price` -> `amount_inr` on flights and hotels.

    A well-trained agent should notice by calling `probe_schema` or by seeing
    that `price` is missing from responses.
    """
    new_field = "amount_inr"
    for f in world.flights:
        f[new_field] = f.pop(world.price_field_flight)
    for h in world.hotels:
        h[new_field] = h.pop(world.price_field_hotel)
    world.price_field_flight = new_field
    world.price_field_hotel = new_field


def apply_change_units(world: WorldState) -> None:
    """Switch prices from rupees to paise (1 rupee = 100 paise).

    Numbers suddenly look 100x larger. An untrained agent that filters by
    `price < 8000` will return nothing; a trained agent should either notice
    the jump or re-read policies.
    """
    for f in world.flights:
        f[world.price_field_flight] = f[world.price_field_flight] * 100
    for h in world.hotels:
        h[world.price_field_hotel] = h[world.price_field_hotel] * 100
    world.price_unit = "paise"


def apply_mutate_enum(world: WorldState) -> None:
    """Change `refundable: true/false` to `refund_tier: "full"|"partial"|"none"`.

    This is the drift that most often causes *silent* policy violations —
    booking a `refund_tier: "none"` item when the user asked for refundable
    and punishing that is the signature SchemaQuake reward signal.
    """
    new_field = "refund_tier"
    for items in (world.flights, world.hotels):
        for it in items:
            was_refundable = bool(it[world.refundable_field])
            it.pop(world.refundable_field)
            it[new_field] = "full" if was_refundable else "none"
    world.refundable_field = new_field
    world.refundable_representation = "enum"


def apply_update_policy(world: WorldState) -> None:
    """Mutate the policies document.

    The cancellation window changes from 24h to 48h and a new clause is
    inserted that changes the interpretation of "refundable" (a honest-to-god
    T&Cs update). Agents that re-read policies after drift are rewarded.
    """
    addendum = (
        "\n\n## 6. Addendum (effective immediately)\n\n"
        "- Cancellation window is **increased from 24 hours to 48 hours** "
        "before departure / check-in. Bookings cancelled within 48h are "
        "non-refundable regardless of the `refundable` flag.\n"
        "- The `refundable` flag now indicates whether the booking is "
        "**eligible** for refund; actual eligibility depends on the 48h "
        "window above.\n"
    )
    # Replace version marker so agents that parse it detect the change.
    world.policies_md = world.policies_md.replace(
        "**Document version:** v1", "**Document version:** v2"
    )
    world.policies_md += addendum


DRIFT_FN = {
    DriftType.RENAME_FIELD: apply_rename_field,
    DriftType.CHANGE_UNITS: apply_change_units,
    DriftType.MUTATE_ENUM: apply_mutate_enum,
    DriftType.UPDATE_POLICY: apply_update_policy,
}


@dataclass
class DriftScheduler:
    """Decides when and which drift fires during an episode.

    Drift fires exactly once. The step is sampled from [min_step, max_step].
    Set drift_type=NONE for a control episode (~20% of training data
    recommended, so the agent doesn't become drift-happy).
    """

    drift_type: DriftType
    drift_step: Optional[int]
    fired: bool = False

    @classmethod
    def random(
        cls,
        rng: random.Random,
        min_step: int = 3,
        max_step: int = 12,
        p_no_drift: float = 0.2,
    ) -> "DriftScheduler":
        if rng.random() < p_no_drift:
            return cls(drift_type=DriftType.NONE, drift_step=None)
        drift = rng.choice([
            DriftType.RENAME_FIELD,
            DriftType.CHANGE_UNITS,
            DriftType.MUTATE_ENUM,
            DriftType.UPDATE_POLICY,
        ])
        step = rng.randint(min_step, max_step)
        return cls(drift_type=drift, drift_step=step)

    def maybe_fire(self, world: WorldState, current_step: int) -> bool:
        if self.fired or self.drift_type == DriftType.NONE:
            return False
        if self.drift_step is not None and current_step >= self.drift_step:
            DRIFT_FN[self.drift_type](world)
            self.fired = True
            return True
        return False
