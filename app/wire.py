from __future__ import annotations

import json
from typing import Any

from app.models import (
    ConversationEntry,
    ConversationResult,
    GenerationRequest,
    ImageAttachment,
    LinkAttachment,
    NoAttachment,
)


class WireFormatError(ValueError):
    pass


def _allowed_keys(item: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = set(item) - allowed
    if extra:
        raise WireFormatError(
            f"{label} contains unsupported keys: {', '.join(sorted(extra))}"
        )


def _parse_attachment(value: Any, request: GenerationRequest):
    if value is None:
        return NoAttachment()

    if not isinstance(value, dict):
        raise WireFormatError("attachment 'a' must be an object")

    kind = value.get("k")

    if kind == "i":
        if not request.features.images:
            raise WireFormatError("image attachment returned while images are disabled")
        _allowed_keys(value, {"k", "d", "x"}, "image attachment")
        description = value.get("d")
        alt_text = value.get("x")
        if not isinstance(description, str) or not isinstance(alt_text, str):
            raise WireFormatError("image attachment requires string fields d and x")
        return ImageAttachment(
            type="image",
            description=description,
            alt_text=alt_text,
        )

    if kind == "l":
        if not request.features.links:
            raise WireFormatError("link attachment returned while links are disabled")
        _allowed_keys(value, {"k", "q", "x"}, "link attachment")
        query = value.get("q")
        label = value.get("x")
        if not isinstance(query, str) or not isinstance(label, str):
            raise WireFormatError("link attachment requires string fields q and x")
        return LinkAttachment(
            type="link",
            query=query,
            label=label,
        )

    raise WireFormatError("attachment kind must be 'i' or 'l'")


def _json_candidates(content: str) -> list[str]:
    """Return conservative JSON candidates without attempting semantic repair.

    Structured-output models occasionally wrap otherwise valid JSON in a
    Markdown fence or a short textual preface. We may safely strip those
    wrappers, but we deliberately do not guess missing braces, quotes or
    values: malformed/truncated JSON still goes through the normal regeneration
    path.
    """
    stripped = content.strip()
    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    add(stripped)

    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            add("\n".join(lines[1:-1]))

    first = stripped.find("{")
    last = stripped.rfind("}")
    if 0 <= first < last:
        add(stripped[first:last + 1])

    return candidates


def parse_wire_result(
    request: GenerationRequest,
    content: str,
) -> ConversationResult:
    raw = None
    last_error: json.JSONDecodeError | None = None
    for candidate in _json_candidates(content):
        try:
            raw = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc

    if raw is None:
        if last_error is None:
            raise WireFormatError("invalid JSON: empty model output")
        raise WireFormatError(f"invalid JSON: {last_error}") from last_error

    if not isinstance(raw, dict):
        raise WireFormatError("wire result root must be an object")

    _allowed_keys(raw, {"e"}, "wire root")

    entries_raw = raw.get("e")
    if not isinstance(entries_raw, list):
        raise WireFormatError("wire root requires array field 'e'")

    entries: list[ConversationEntry] = []

    for index, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            raise WireFormatError(f"entry {index} must be an object")

        if "n" in item:
            _allowed_keys(item, {"n"}, f"narration entry {index}")
            text = item.get("n")
            if not isinstance(text, str):
                raise WireFormatError(f"narration entry {index} requires string field n")

            entries.append(
                ConversationEntry(
                    type="narration",
                    speaker_id="narrator",
                    text=text,
                    typing_indicator=False,
                    attachment=NoAttachment(),
                )
            )
            continue

        _allowed_keys(item, {"s", "t", "a"}, f"message entry {index}")

        speaker = item.get("s")
        text = item.get("t")

        if not isinstance(speaker, str):
            raise WireFormatError(f"message entry {index} requires string field s")
        if not isinstance(text, str):
            raise WireFormatError(f"message entry {index} requires string field t")

        attachment = _parse_attachment(item.get("a"), request)

        entries.append(
            ConversationEntry(
                type="message",
                speaker_id=speaker,
                text=text,
                typing_indicator=False,
                attachment=attachment,
            )
        )

    return ConversationResult(
        schema_version="1.0",
        language=request.language,
        entries=entries,
    )


def result_to_wire_dict(
    result: ConversationResult,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    for entry in result.entries:
        if entry.type == "narration":
            entries.append({"n": entry.text})
            continue

        item: dict[str, Any] = {
            "s": entry.speaker_id,
            "t": entry.text,
        }

        if isinstance(entry.attachment, ImageAttachment):
            item["a"] = {
                "k": "i",
                "d": entry.attachment.description,
                "x": entry.attachment.alt_text,
            }
        elif isinstance(entry.attachment, LinkAttachment):
            item["a"] = {
                "k": "l",
                "q": entry.attachment.query,
                "x": entry.attachment.label,
            }

        entries.append(item)

    return {"e": entries}


def result_to_wire_json(result: ConversationResult) -> str:
    return json.dumps(
        result_to_wire_dict(result),
        ensure_ascii=False,
        separators=(",", ":"),
    )
