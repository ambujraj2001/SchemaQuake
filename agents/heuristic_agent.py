"""A drift-aware heuristic agent.

Hand-coded upper bound. Demonstrates that the environment is *solvable*
and sets a ceiling that the trained LLM should approach.

Strategy:
  1. Read policies once up front.
  2. Search flights with user constraints.
  3. Before booking, look at the response structure. If the expected field
     names are missing, probe_schema and re-interpret prices.
  4. Filter to refundable (under whichever representation is active) and
     under max price (canonical rupees). Pick the cheapest.
  5. Book, then submit.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemaquake.types import SQAction, SQObservation, ToolName


EXPECTED_FLIGHT_PRICE_FIELD = "price"
EXPECTED_REFUNDABLE_FIELD = "refundable"


class HeuristicAgent:
    def __init__(self) -> None:
        self._reset_internal()

    def _reset_internal(self) -> None:
        self._brief: Optional[Dict[str, Any]] = None
        self._schema_probed: bool = False
        self._policies_read: bool = False
        self._search_results: List[Dict[str, Any]] = []
        self._active_schema: Dict[str, Any] = {
            "flight_price_field": EXPECTED_FLIGHT_PRICE_FIELD,
            "hotel_price_field": "price_per_night",
            "refundable_field": EXPECTED_REFUNDABLE_FIELD,
            "price_unit": "rupees",
            "refundable_representation": "bool",
        }
        self._last_booking_id: Optional[str] = None
        self._phase: str = "read_policies"

    def reset(self) -> None:
        self._reset_internal()

    def act(self, obs: SQObservation) -> SQAction:
        if obs.episode_brief is not None:
            self._brief = obs.episode_brief
            self._phase = "read_policies"

        # Absorb any tool response.
        tr = obs.tool_result or {}
        if "policies_md" in tr:
            self._policies_read = True
            self._phase = "search"
        if "flight_price_field" in tr:
            self._active_schema.update(tr)
            self._schema_probed = True
            self._phase = "search" if not self._search_results else "choose"
        if "results" in tr and tr.get("results") is not None:
            self._search_results = tr["results"]
            # Check structural drift: if any result is missing the expected
            # price field, probe before proceeding.
            if self._search_results and self._needs_probe(self._search_results[0]):
                self._phase = "probe"
            else:
                self._phase = "choose"
        if tr.get("status") == "confirmed":
            self._last_booking_id = tr["booking_id"]
            self._phase = "submit"

        return self._plan_next()

    def _needs_probe(self, sample: Dict[str, Any]) -> bool:
        expected_price = self._active_schema["flight_price_field"]
        expected_refund = self._active_schema["refundable_field"]
        return (expected_price not in sample) or (expected_refund not in sample)

    def _plan_next(self) -> SQAction:
        assert self._brief is not None, "Brief not yet received."
        b = self._brief

        if self._phase == "read_policies":
            return SQAction(tool=ToolName.READ_POLICIES, confidence=0.9)

        if self._phase == "probe":
            return SQAction(tool=ToolName.PROBE_SCHEMA, confidence=0.7)

        if self._phase == "search":
            return SQAction(
                tool=ToolName.SEARCH_FLIGHTS,
                args={
                    "origin": b["origin"],
                    "destination": b["destination"],
                },
                confidence=0.85,
            )

        if self._phase == "choose":
            pick = self._pick_best_flight()
            if pick is None:
                # No valid option — ask the user what to relax.
                return SQAction(
                    tool=ToolName.ASK_USER,
                    args={"question": "No flights meet your constraints; relax price or refundability?"},
                    confidence=0.4,
                )
            return SQAction(
                tool=ToolName.BOOK, args={"item_id": pick["id"]}, confidence=0.9
            )

        if self._phase == "submit":
            return SQAction(
                tool=ToolName.SUBMIT,
                args={"booking_id": self._last_booking_id},
                confidence=0.95,
            )

        # Defensive fallback.
        return SQAction(tool=ToolName.NOOP, confidence=0.5)

    def _pick_best_flight(self) -> Optional[Dict[str, Any]]:
        price_field = self._active_schema["flight_price_field"]
        refund_field = self._active_schema["refundable_field"]
        unit = self._active_schema["price_unit"]
        rep = self._active_schema["refundable_representation"]

        def canonical_price(item: Dict[str, Any]) -> int:
            raw = item.get(price_field, 10**9)
            return raw // 100 if unit == "paise" else raw

        def refundable(item: Dict[str, Any]) -> bool:
            v = item.get(refund_field)
            if rep == "bool":
                return bool(v)
            return str(v).lower() in ("full", "partial")

        b = self._brief
        candidates = [
            it for it in self._search_results
            if canonical_price(it) <= b["max_price_rupees"]
            and (not b["must_be_refundable"] or refundable(it))
        ]
        if not candidates:
            return None
        return min(candidates, key=canonical_price)
