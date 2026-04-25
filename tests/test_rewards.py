from schemaquake.rewards import RewardEngine
from schemaquake.types import DriftType, EpisodeBrief, GroundTruth, ToolName
from schemaquake.world import WorldState


def _brief(max_price=8000, refund=True):
    return EpisodeBrief(
        request_text="",
        origin="BLR",
        destination="DEL",
        max_price_rupees=max_price,
        must_be_refundable=refund,
    )


def test_drift_detection_within_window():
    gt = GroundTruth(drift_type=DriftType.RENAME_FIELD, drift_step=3)
    engine = RewardEngine(_brief(), gt)
    world = WorldState.fresh()
    delta = engine.on_step(
        tool=ToolName.PROBE_SCHEMA, tool_args={}, tool_result={},
        confidence=0.5, current_step=3, drift_fired_this_step=True, world=world,
    )
    assert engine.state.breakdown.drift > 0
    assert gt.drift_detected_at == 3


def test_drift_detection_outside_window_no_bonus():
    gt = GroundTruth(drift_type=DriftType.RENAME_FIELD, drift_step=3)
    engine = RewardEngine(_brief(), gt)
    world = WorldState.fresh()
    engine.on_step(
        tool=ToolName.PROBE_SCHEMA, tool_args={}, tool_result={},
        confidence=0.5, current_step=10, drift_fired_this_step=False, world=world,
    )
    assert engine.state.breakdown.drift == 0


def test_silent_violation_penalty():
    gt = GroundTruth(drift_type=DriftType.NONE)
    engine = RewardEngine(_brief(refund=True), gt)
    world = WorldState.fresh()
    engine.on_step(
        tool=ToolName.BOOK, tool_args={},
        tool_result={"status": "confirmed", "price_charged": 3900, "refundable": False},
        confidence=0.9, current_step=5, drift_fired_this_step=False, world=world,
    )
    assert engine.state.breakdown.violation == -1.0
    assert gt.silent_violation is True


def test_hedge_reward_for_calibrated_ask():
    engine = RewardEngine(_brief(), GroundTruth())
    world = WorldState.fresh()
    engine.on_step(
        tool=ToolName.ASK_USER, tool_args={"question": "?"}, tool_result={},
        confidence=0.3, current_step=4, drift_fired_this_step=False, world=world,
    )
    assert engine.state.breakdown.hedge == 0.1


def test_hedge_penalty_for_overconfident_ask():
    engine = RewardEngine(_brief(), GroundTruth())
    world = WorldState.fresh()
    engine.on_step(
        tool=ToolName.ASK_USER, tool_args={"question": "?"}, tool_result={},
        confidence=0.95, current_step=4, drift_fired_this_step=False, world=world,
    )
    assert engine.state.breakdown.hedge == -0.1


def test_terminal_task_reward_success():
    brief = _brief(max_price=8000, refund=True)
    engine = RewardEngine(brief, GroundTruth())
    world = WorldState.fresh()
    world.bookings["CNF-1"] = {
        "booking_id": "CNF-1", "kind": "flight", "item_id": "F-101",
        "canonical_price_rupees": 5400, "refundable": True,
    }
    bd = engine.on_terminal("CNF-1", world)
    assert bd.task >= 1.0


def test_terminal_task_reward_partial():
    brief = _brief(max_price=4000, refund=True)  # too low
    engine = RewardEngine(brief, GroundTruth())
    world = WorldState.fresh()
    world.bookings["CNF-1"] = {
        "booking_id": "CNF-1", "kind": "flight", "item_id": "F-101",
        "canonical_price_rupees": 5400, "refundable": True,
    }
    bd = engine.on_terminal("CNF-1", world)
    assert 0 < bd.task < 1.0
