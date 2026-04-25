"""LLM-backed agent that parses a single tool-call JSON per turn.

Supports any OpenAI-compatible chat endpoint (OpenAI, Together, HF TGI,
vLLM, etc.) via the OpenAI Python SDK. The base URL and model name are
configured via environment variables:

  OPENAI_API_KEY
  OPENAI_BASE_URL   (optional, defaults to https://api.openai.com/v1)
  OPENAI_MODEL      (default gpt-4o-mini)

The agent keeps its own conversation history scoped to a single episode and
resets it via `reset()`.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from schemaquake.prompts import SYSTEM_PROMPT
from schemaquake.types import SQAction, SQObservation, ToolName


class LLMAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        try:
            from openai import OpenAI  # lazy import
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "LLMAgent requires `pip install openai`. "
                "Run `pip install -e .[agents]`."
            ) from e

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._temperature = temperature
        self._history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def act(self, obs: SQObservation) -> SQAction:
        if not self._history:
            self.reset()

        user_turn = self._format_observation(obs)
        self._history.append({"role": "user", "content": user_turn})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=self._history,
            temperature=self._temperature,
        )
        content = resp.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": content})

        return self._parse(content)

    def _format_observation(self, obs: SQObservation) -> str:
        if obs.episode_brief is not None:
            return (
                "NEW EPISODE\n"
                f"User brief: {json.dumps(obs.episode_brief, indent=2)}\n"
                "Respond with one JSON tool call."
            )
        return (
            f"Step result:\n{json.dumps(obs.tool_result, default=str)[:4000]}\n"
            "Respond with one JSON tool call."
        )

    def _parse(self, content: str) -> SQAction:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            return SQAction(tool=ToolName.NOOP, confidence=0.0)
        try:
            data = json.loads(m.group(0))
            tool_raw = data.get("tool", "noop")
            return SQAction(
                tool=ToolName(tool_raw),
                args=data.get("args") or {},
                confidence=data.get("confidence"),
            )
        except Exception:
            return SQAction(tool=ToolName.NOOP, confidence=0.0)
