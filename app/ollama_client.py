from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class OllamaResult:
    content: str
    prompt_tokens: int | None
    generated_tokens: int | None
    load_duration_ns: int | None
    prompt_eval_duration_ns: int | None
    eval_duration_ns: int | None
    total_duration_ns: int | None
    thinking: str | None = None
    done_reason: str | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.generated_tokens or not self.eval_duration_ns:
            return None
        seconds = self.eval_duration_ns / 1_000_000_000
        if seconds <= 0:
            return None
        return self.generated_tokens / seconds


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds)

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float,
        num_ctx: int,
        num_predict: int,
        keep_alive: str,
        think: bool | str | None = None,
    ) -> OllamaResult:
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": keep_alive,
            "format": schema,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        if think is not None:
            payload["think"] = think

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return OllamaResult(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count"),
            generated_tokens=data.get("eval_count"),
            load_duration_ns=data.get("load_duration"),
            prompt_eval_duration_ns=data.get("prompt_eval_duration"),
            eval_duration_ns=data.get("eval_duration"),
            total_duration_ns=data.get("total_duration"),
            thinking=data.get("message", {}).get("thinking"),
            done_reason=data.get("done_reason"),
        )

    async def warmup(
        self,
        *,
        model: str,
        keep_alive: str,
        num_ctx: int,
    ) -> OllamaResult:
        """
        Load the model into memory using a one-token generate request.
        The generated content is discarded; keep_alive keeps the model resident.
        """
        payload = {
            "model": model,
            "prompt": " ",
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "num_predict": 1,
                "num_ctx": num_ctx,
                "temperature": 0.0,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return OllamaResult(
            content=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count"),
            generated_tokens=data.get("eval_count"),
            load_duration_ns=data.get("load_duration"),
            prompt_eval_duration_ns=data.get("prompt_eval_duration"),
            eval_duration_ns=data.get("eval_duration"),
            total_duration_ns=data.get("total_duration"),
            thinking=data.get("thinking"),
            done_reason=data.get("done_reason"),
        )

    async def tags(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()

    async def version(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/version")
            response.raise_for_status()
            return response.json()
