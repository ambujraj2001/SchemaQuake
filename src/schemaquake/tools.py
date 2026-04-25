"""Pure functions implementing each tool.

Every tool takes the current `WorldState` (mutated in place by drift operators)
and returns a structured result dict. These functions never raise; they encode
errors as `{"error": ...}` so the agent can read them as observations.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .world import WorldState


def _project(item: Dict[str, Any], world: WorldState, kind: str) -> Dict[str, Any]:
    """Return a *drift-visible* projection of a flight/hotel item.

    The agent sees whichever field names / units / enum types the drift
    operators have currently applied. This is the whole point of SchemaQuake:
    the visible surface changes, the semantics underneath do not.
    """
    out = dict(item)  # shallow copy is enough — caller won't mutate
    return out


def search_flights(
    world: WorldState,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    max_price: Optional[int] = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for f in world.flights:
        if origin and f["from"] != origin:
            continue
        if destination and f["to"] != destination:
            continue
        if max_price is not None:
            canon = world.price_of_flight(f)
            if canon > max_price:
                continue
        results.append(_project(f, world, "flight"))
    return {
        "schema_version": "v1" if world.price_field_flight == "price" else "v2",
        "price_field": world.price_field_flight,
        "refundable_field": world.refundable_field,
        "results": results,
    }


def search_hotels(
    world: WorldState,
    city: Optional[str] = None,
    max_price_per_night: Optional[int] = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for h in world.hotels:
        if city and h["city"] != city:
            continue
        if max_price_per_night is not None:
            canon = world.price_of_hotel(h)
            if canon > max_price_per_night:
                continue
        results.append(_project(h, world, "hotel"))
    return {
        "schema_version": "v1" if world.price_field_hotel == "price_per_night" else "v2",
        "price_field": world.price_field_hotel,
        "refundable_field": world.refundable_field,
        "results": results,
    }


def read_policies(world: WorldState) -> Dict[str, Any]:
    return {"policies_md": world.policies_md}


def probe_schema(world: WorldState) -> Dict[str, Any]:
    """Cheap introspection tool. Returns the *current* schema descriptor.

    Agents are expected to call this when they see unexpected data.
    Calling it regularly is fine but costs a small reward penalty to avoid
    degenerate "always probe" policies.
    """
    return {
        "flight_price_field": world.price_field_flight,
        "hotel_price_field": world.price_field_hotel,
        "refundable_field": world.refundable_field,
        "price_unit": world.price_unit,
        "refundable_representation": world.refundable_representation,
    }


def _find_by_id(items: List[Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
    for it in items:
        if it.get("id") == item_id:
            return it
    return None


def book(world: WorldState, item_id: str) -> Dict[str, Any]:
    flight = _find_by_id(world.flights, item_id)
    hotel = _find_by_id(world.hotels, item_id)
    item = flight or hotel
    if item is None:
        return {"error": "item_not_found", "item_id": item_id}

    kind = "flight" if flight is not None else "hotel"
    if kind == "flight" and item.get("seats_left", 0) <= 0:
        return {"error": "sold_out", "item_id": item_id}
    if kind == "hotel" and item.get("rooms_left", 0) <= 0:
        return {"error": "sold_out", "item_id": item_id}

    price = (
        world.price_of_flight(item) if kind == "flight" else world.price_of_hotel(item)
    )
    refundable = world.is_refundable(item)
    booking_id = f"CNF-{uuid.uuid4().hex[:8].upper()}"
    world.bookings[booking_id] = {
        "booking_id": booking_id,
        "kind": kind,
        "item_id": item_id,
        "canonical_price_rupees": price,
        "refundable": refundable,
    }
    # Decrement inventory so a second book() on the same item can fail realistically.
    if kind == "flight":
        item["seats_left"] -= 1
    else:
        item["rooms_left"] -= 1
    return {
        "status": "confirmed",
        "booking_id": booking_id,
        "kind": kind,
        "item_id": item_id,
        "price_charged": price,
        "refundable": refundable,
    }


def cancel(world: WorldState, booking_id: str) -> Dict[str, Any]:
    rec = world.bookings.get(booking_id)
    if rec is None:
        return {"error": "booking_not_found", "booking_id": booking_id}
    if not rec["refundable"]:
        return {"status": "denied", "reason": "non_refundable", "booking_id": booking_id}
    rec["cancelled"] = True
    return {"status": "refunded", "booking_id": booking_id}


def ask_user(world: WorldState, question: str) -> Dict[str, Any]:
    """Simulated user response.

    The user's real preference is encoded in the EpisodeBrief; here we return a
    templated confirmation that mirrors the brief. In training this is fine —
    the agent just needs to learn *when* to ask, not what the user will say.
    """
    return {
        "user_response": "Please follow the original request exactly as stated.",
        "question_echo": question,
    }


TOOL_REGISTRY = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "read_policies": read_policies,
    "probe_schema": probe_schema,
    "book": book,
    "cancel": cancel,
    "ask_user": ask_user,
}
