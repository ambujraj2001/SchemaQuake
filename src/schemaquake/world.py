"""Mutable world state: flights, hotels, and the policies document.

The world is reloaded from packaged JSON/Markdown on every reset so that a
DriftOperator can freely mutate it during an episode without corrupting
subsequent episodes.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Dict, List


def _load_json(filename: str) -> Dict[str, Any]:
    with resources.files("schemaquake.data").joinpath(filename).open("r") as f:
        return json.load(f)


def _load_text(filename: str) -> str:
    with resources.files("schemaquake.data").joinpath(filename).open("r") as f:
        return f.read()


@dataclass
class WorldState:
    """Mutable per-episode world state.

    Fields are deep-copied on construction so drift mutations are scoped to
    a single episode.
    """

    flights: List[Dict[str, Any]]
    hotels: List[Dict[str, Any]]
    policies_md: str

    # Schema metadata — these can be mutated by drift operators.
    price_field_flight: str = "price"
    price_field_hotel: str = "price_per_night"
    refundable_field: str = "refundable"
    price_unit: str = "rupees"       # "rupees" or "paise"
    refundable_representation: str = "bool"  # "bool" or "enum"

    # Bookings created during the episode (booking_id -> record).
    bookings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> "WorldState":
        flights_doc = _load_json("flights.json")
        hotels_doc = _load_json("hotels.json")
        policies_md = _load_text("policies.md")
        return cls(
            flights=copy.deepcopy(flights_doc["flights"]),
            hotels=copy.deepcopy(hotels_doc["hotels"]),
            policies_md=policies_md,
        )

    def price_of_flight(self, flight: Dict[str, Any]) -> int:
        """Always returns the canonical price in rupees regardless of drift.

        Used by the hidden reward function; the agent sees the possibly-drifted
        field directly.
        """
        raw = flight[self.price_field_flight]
        return raw // 100 if self.price_unit == "paise" else raw

    def price_of_hotel(self, hotel: Dict[str, Any]) -> int:
        raw = hotel[self.price_field_hotel]
        return raw // 100 if self.price_unit == "paise" else raw

    def is_refundable(self, item: Dict[str, Any]) -> bool:
        """Decode refundability through drift-aware representation."""
        val = item[self.refundable_field]
        if self.refundable_representation == "bool":
            return bool(val)
        # enum mode
        return str(val).lower() in ("full", "partial")
