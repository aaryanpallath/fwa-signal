"""LLM access through a local Ollama server (no API key, no data leaves the machine)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class Completion:
    text: str
    latency_s: float
    output_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, prompt: str, json_mode: bool = False) -> Completion: ...


@dataclass
class OllamaLLM:
    model: str = field(default_factory=lambda: os.environ.get("FWA_LLM_MODEL", "gemma4"))
    url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    timeout_s: float = 180

    def complete(self, system: str, prompt: str, json_mode: bool = False) -> Completion:
        t = time.monotonic()
        body = {"model": self.model, "stream": False, "options": {"temperature": 0, "seed": 7},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["format"] = "json"
        r = httpx.post(f"{self.url}/api/chat", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        d = r.json()
        return Completion(d["message"]["content"], time.monotonic() - t, int(d.get("eval_count", 0)))


@dataclass
class ScriptedLLM:
    """Test double: returns canned replies in order and records what it was asked."""
    replies: list[str]
    prompts: list[str] = field(default_factory=list)

    def complete(self, system: str, prompt: str, json_mode: bool = False) -> Completion:
        self.prompts.append(prompt)
        return Completion(self.replies.pop(0), 0.0)
