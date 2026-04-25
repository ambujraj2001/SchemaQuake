"""A random baseline that roughly follows a search->book->submit template.

This is a *weak* baseline on purpose: it establishes the floor of the reward
distribution so the improvement curve of a trained LLM is unambiguous.
"""
from __future__ import annotations

import random
from typing import List

from schemaquake.types import SQAction, SQObservation, ToolName


class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._plan: List[SQAction] = []
        self._last_search_ids: List[str] = []

    def reset(self) -> None:
        self._plan = []
        self._last_search_ids = []

    def act(self, obs: SQObservation) -> SQAction:
        # Episode start: seed a plan.
        if obs.episode_brief is not None:
            brief = obs.episode_brief
            self._plan = [
                SQAction(
                    tool=ToolName.SEARCH_FLIGHTS,
                    args={
                        "origin": brief["origin"],
                        "destination": brief["destination"],
                        "max_price": brief["max_price_rupees"],
                    },
                    confidence=0.8,
                ),
            ]

        # Consume from search results if available.
        if obs.tool_result.get("results"):
            self._last_search_ids = [
                r.get("id") for r in obs.tool_result["results"] if r.get("id")
            ]
            if self._last_search_ids:
                chosen = self._rng.choice(self._last_search_ids)
                return SQAction(
                    tool=ToolName.BOOK, args={"item_id": chosen}, confidence=0.8
                )

        if obs.tool_result.get("status") == "confirmed":
            return SQAction(
                tool=ToolName.SUBMIT,
                args={"booking_id": obs.tool_result["booking_id"]},
                confidence=0.9,
            )

        if self._plan:
            return self._plan.pop(0)

        # Fallback: end the episode.
        return SQAction(tool=ToolName.SUBMIT, args={}, confidence=0.5)
