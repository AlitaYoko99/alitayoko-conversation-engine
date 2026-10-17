from __future__ import annotations

from typing import Any

from app.models import GenerationRequest


def _attachment_schema(request: GenerationRequest) -> dict[str, Any] | None:
    variants: list[dict[str, Any]] = []

    if request.features.images:
        variants.append(
            {
                "type": "object",
                "properties": {
                    "k": {"type": "string", "const": "i"},
                    "d": {"type": "string", "minLength": 3, "maxLength": 500},
                    "x": {"type": "string", "minLength": 1, "maxLength": 250},
                },
                "required": ["k", "d", "x"],
                "additionalProperties": False,
            }
        )

    if request.features.links:
        variants.append(
            {
                "type": "object",
                "properties": {
                    "k": {"type": "string", "const": "l"},
                    "q": {"type": "string", "minLength": 3, "maxLength": 180},
                    "x": {"type": "string", "minLength": 2, "maxLength": 80},
                },
                "required": ["k", "q", "x"],
                "additionalProperties": False,
            }
        )

    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    return {"oneOf": variants}


def build_conversation_schema(request: GenerationRequest) -> dict[str, Any]:
    participant_ids = [participant.id for participant in request.participants]

    message_properties: dict[str, Any] = {
        "s": {
            "type": "string",
            "enum": participant_ids,
        },
        "t": {
            "type": "string",
            "minLength": 1,
        },
    }

    attachment = _attachment_schema(request)
    if attachment is not None:
        message_properties["a"] = attachment

    message_schema: dict[str, Any] = {
        "type": "object",
        "properties": message_properties,
        "required": ["s", "t"],
        "additionalProperties": False,
    }

    entry_variants: list[dict[str, Any]] = [message_schema]

    if request.features.narration and request.limits.max_narrations > 0:
        entry_variants.append(
            {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "string",
                        "minLength": 1,
                    }
                },
                "required": ["n"],
                "additionalProperties": False,
            }
        )

    entry_schema: dict[str, Any]
    if len(entry_variants) == 1:
        entry_schema = entry_variants[0]
    else:
        entry_schema = {"oneOf": entry_variants}

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "e": {
                "type": "array",
                "minItems": request.entry_count,
                "maxItems": request.entry_count,
                "items": entry_schema,
            }
        },
        "required": ["e"],
        "additionalProperties": False,
    }
