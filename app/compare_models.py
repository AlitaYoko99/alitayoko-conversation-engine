from __future__ import annotations

import argparse
import asyncio
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.engine import ConversationEngine, GenerationFailed
from app.model_profiles import get_profile
from app.models import GenerationRequest, GenerationResponse


DEFAULT_PROFILES = ("ministral-14b", "gpt-oss-20b")


def _json_dump(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_request(path: Path) -> GenerationRequest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GenerationRequest.model_validate(raw)


def _installed_names(tags: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in tags.get("models", []):
        name = item.get("name")
        model = item.get("model")
        if isinstance(name, str):
            names.add(name)
        if isinstance(model, str):
            names.add(model)
    return names


def _technical_meta(response: GenerationResponse) -> dict[str, Any]:
    data = response.meta.model_dump()
    # The blind metrics file intentionally omits model identity.
    data.pop("model", None)
    data.pop("model_profile", None)
    return data


async def compare_one(
    request_path: Path,
    output_dir: Path,
    profile_ids: tuple[str, str],
) -> Path:
    request = _load_request(request_path)
    engine = ConversationEngine()

    tags = await engine.client.tags()
    installed = _installed_names(tags)

    profiles = [get_profile(profile_id) for profile_id in profile_ids]
    missing = [
        profile.model
        for profile in profiles
        if profile.model not in installed
        and not any(
            name.startswith(profile.model + ":")
            for name in installed
        )
    ]
    if missing:
        raise RuntimeError(
            "Missing Ollama model(s): " + ", ".join(missing)
        )

    raw_results: list[tuple[Any, GenerationResponse, float]] = []

    for profile in profiles:
        warm = await engine.client.warmup(
            model=profile.model,
            keep_alive=settings.keep_alive,
            num_ctx=settings.num_ctx,
        )
        warm_seconds = (
            warm.total_duration_ns / 1_000_000_000
            if warm.total_duration_ns
            else 0.0
        )

        try:
            response = await engine.generate(
                request,
                profile=profile,
            )
        except GenerationFailed as exc:
            failure_dir = output_dir / (
                "failed-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + "-"
                + profile.id
            )
            failure_dir.mkdir(parents=True, exist_ok=True)

            diagnostic = {
                "profile_id": profile.id,
                "model": profile.model,
                "reasoning_effort": profile.reasoning_effort,
                "ollama_think": profile.ollama_think,
                "num_predict_multiplier": profile.num_predict_multiplier,
                "force_schema_grounding": profile.force_schema_grounding,
                "source_request": request_path.name,
                "request": request.model_dump(),
                "error": str(exc),
                "raw_output": exc.raw_output,
                "violations": [
                    item.model_dump()
                    for item in exc.violations
                ],
            }
            _json_dump(failure_dir / "failure.json", diagnostic)

            raise RuntimeError(
                f"{profile.id} generation failed: {exc}. "
                f"Diagnostic: {failure_dir / 'failure.json'}"
            ) from exc

        raw_results.append((profile, response, warm_seconds))

    labels = ["A", "B"]
    if secrets.randbelow(2):
        raw_results.reverse()

    comparison_id = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + secrets.token_hex(3)
    )
    run_dir = output_dir / comparison_id
    run_dir.mkdir(parents=True, exist_ok=False)

    blind = {
        "comparison_id": comparison_id,
        "source_request": request_path.name,
        "request": request.model_dump(),
        "candidates": {},
        "rating_sheet": {
            "preferred_candidate": None,
            "naturalness": {"A": None, "B": None},
            "coherence": {"A": None, "B": None},
            "language_quality": {"A": None, "B": None},
            "character_consistency": {"A": None, "B": None},
            "publishability": {"A": None, "B": None},
            "notes": "",
        },
    }

    metrics = {
        "comparison_id": comparison_id,
        "source_request": request_path.name,
        "candidates": {},
    }

    key = {
        "comparison_id": comparison_id,
        "source_request": request_path.name,
        "mapping": {},
    }

    for label, (profile, response, warm_seconds) in zip(labels, raw_results):
        blind["candidates"][label] = response.conversation.model_dump()
        metrics["candidates"][label] = {
            "warmup_seconds": warm_seconds,
            **_technical_meta(response),
        }
        key["mapping"][label] = {
            "profile_id": profile.id,
            "label": profile.label,
            "model": profile.model,
            "reasoning_effort": profile.reasoning_effort,
        }

    _json_dump(run_dir / "blind.json", blind)
    _json_dump(run_dir / "metrics.json", metrics)
    _json_dump(run_dir / "key.json", key)

    return run_dir


async def run_suite(
    suite_dir: Path,
    output_dir: Path,
    profile_ids: tuple[str, str],
) -> list[Path]:
    request_files = sorted(suite_dir.glob("*.json"))
    if not request_files:
        raise RuntimeError(f"No JSON requests found in {suite_dir}")

    results: list[Path] = []
    for request_path in request_files:
        results.append(
            await compare_one(
                request_path=request_path,
                output_dir=output_dir,
                profile_ids=profile_ids,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blind A/B comparison for AlitaYoko Conversation Engine model profiles."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--request",
        type=Path,
        help="One GenerationRequest JSON file.",
    )
    source.add_argument(
        "--suite",
        type=Path,
        help="Directory containing GenerationRequest JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluations"),
        help="Output directory (default: evaluations).",
    )
    parser.add_argument(
        "--profiles",
        nargs=2,
        default=list(DEFAULT_PROFILES),
        metavar=("PROFILE_A", "PROFILE_B"),
        help="Exactly two registered model profile IDs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    profile_ids = (args.profiles[0], args.profiles[1])
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        if args.request:
            paths = [
                asyncio.run(
                    compare_one(
                        request_path=args.request,
                        output_dir=args.out,
                        profile_ids=profile_ids,
                    )
                )
            ]
        else:
            paths = asyncio.run(
                run_suite(
                    suite_dir=args.suite,
                    output_dir=args.out,
                    profile_ids=profile_ids,
                )
            )
    except (ValidationError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
        return

    print("Comparison complete.")
    for path in paths:
        print(path)
        print(f"  Blind:   {path / 'blind.json'}")
        print(f"  Metrics: {path / 'metrics.json'}")
        print(f"  Key:     {path / 'key.json'}")


if __name__ == "__main__":
    main()
