import random

from schemaquake.drift import (
    DriftScheduler,
    apply_change_units,
    apply_mutate_enum,
    apply_rename_field,
    apply_update_policy,
)
from schemaquake.tools import search_flights
from schemaquake.types import DriftType
from schemaquake.world import WorldState


def test_rename_field_drift():
    world = WorldState.fresh()
    apply_rename_field(world)
    res = search_flights(world, origin="BLR", destination="DEL", max_price=10000)
    sample = res["results"][0]
    assert "price" not in sample
    assert "amount_inr" in sample
    assert world.price_field_flight == "amount_inr"


def test_change_units_drift():
    world = WorldState.fresh()
    before = world.flights[0]["price"]
    apply_change_units(world)
    after = world.flights[0]["price"]
    assert after == before * 100
    assert world.price_unit == "paise"
    assert world.price_of_flight(world.flights[0]) == before


def test_mutate_enum_drift():
    world = WorldState.fresh()
    apply_mutate_enum(world)
    f = world.flights[0]
    assert "refundable" not in f
    assert f["refund_tier"] in ("full", "none")
    assert world.refundable_representation == "enum"
    assert world.is_refundable(f) == (f["refund_tier"] == "full")


def test_update_policy_drift():
    world = WorldState.fresh()
    apply_update_policy(world)
    assert "48 hours" in world.policies_md
    assert "Document version:** v2" in world.policies_md


def test_scheduler_fires_once():
    rng = random.Random(42)
    sched = DriftScheduler(drift_type=DriftType.RENAME_FIELD, drift_step=3)
    world = WorldState.fresh()
    fired_at = [sched.maybe_fire(world, step) for step in range(1, 8)]
    assert fired_at.count(True) == 1
    assert fired_at[2] is True  # step 3, zero-indexed position 2


def test_scheduler_no_drift():
    sched = DriftScheduler(drift_type=DriftType.NONE, drift_step=None)
    world = WorldState.fresh()
    assert not any(sched.maybe_fire(world, s) for s in range(1, 20))


def test_scheduler_random_respects_p_no_drift():
    rng = random.Random(0)
    out = [DriftScheduler.random(rng, p_no_drift=1.0).drift_type for _ in range(10)]
    assert all(d == DriftType.NONE for d in out)
