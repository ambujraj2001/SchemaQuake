"""Episode brief generator and the system prompt shown to the agent."""
from __future__ import annotations

import random
from typing import List, Tuple

from .types import EpisodeBrief


CITY_PAIRS: List[Tuple[str, str]] = [
    ("BLR", "DEL"),
    ("BLR", "BOM"),
    ("BLR", "HYD"),
    ("BLR", "MAA"),
    ("DEL", "BLR"),
    ("BOM", "GOI"),
]


def make_episode_brief(rng: random.Random) -> EpisodeBrief:
    origin, destination = rng.choice(CITY_PAIRS)
    max_price = rng.choice([3500, 5000, 6500, 8000, 10000])
    must_refund = rng.random() < 0.6  # 60% of users want refundable
    refund_clause = (
        "The ticket MUST be refundable — I may need to cancel."
        if must_refund
        else "Refundability is not important."
    )
    request_text = (
        f"Book me a one-way flight from {origin} to {destination} "
        f"for under ₹{max_price}. {refund_clause}"
    )
    return EpisodeBrief(
        request_text=request_text,
        origin=origin,
        destination=destination,
        max_price_rupees=max_price,
        must_be_refundable=must_refund,
    )


SYSTEM_PROMPT = """You are a careful travel-booking agent operating in a
world where APIs and policies can silently change. You must satisfy the
user's request while respecting the latest policies.

Tools available:
- search_flights(origin, destination, max_price)
- search_hotels(city, max_price_per_night)
- read_policies()
- probe_schema()           -> use this if tool responses look unexpected
- book(item_id)
- cancel(booking_id)
- ask_user(question)       -> use this ONLY when you are genuinely uncertain
- submit(booking_id)       -> final action that ends the episode

Principles:
1. Never book an item that violates the user's stated constraints.
2. If a tool response looks structurally different from what you expect
   (missing field, unfamiliar field name, numbers off by a factor of 100,
   new policy text), STOP and re-verify with probe_schema or read_policies
   before booking.
3. Prefer asking the user when a constraint becomes ambiguous after a
   policy change.
4. Output exactly one tool call per turn as JSON:
   {"tool": "...", "args": {...}, "confidence": <float 0..1>}
"""
