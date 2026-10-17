from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app import __version__
from app.config import settings
from app.engine import ConversationEngine, GenerationFailed
from app.model_profiles import list_profiles
from app.models import GenerationRequest, GenerationResponse
from app.schema_builder import build_conversation_schema
from app.research import ResearchError, ResearchUnavailable


logger = logging.getLogger("alitayoko.engine")
engine = ConversationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.prewarm_on_start:
        try:
            result = await engine.client.warmup(
                model=settings.ollama_model,
                keep_alive=settings.keep_alive,
                num_ctx=settings.num_ctx,
            )
            load_seconds = (
                result.load_duration_ns / 1_000_000_000
                if result.load_duration_ns
                else 0.0
            )
            logger.info(
                "Prewarmed %s (load %.2fs, keep_alive=%s)",
                settings.ollama_model,
                load_seconds,
                settings.keep_alive,
            )
        except Exception:
            # The API should still start if Ollama is temporarily unavailable.
            logger.exception("Ollama prewarm failed")

    yield


app = FastAPI(
    title="AlitaYoko Conversation Engine",
    version=__version__,
    description="Local AI backend for the AlitaYoko Conversations WordPress plugin.",
    lifespan=lifespan,
)


def require_api_key(
    x_ay_api_key: str | None = Header(default=None),
) -> None:
    if not settings.api_key:
        return

    if x_ay_api_key is None or not hmac.compare_digest(
        x_ay_api_key,
        settings.api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.get(
    "/v1/model-profiles",
    dependencies=[Depends(require_api_key)],
)
async def model_profiles() -> dict:
    return {
        "profiles": [
            {
                "id": profile.id,
                "model": profile.model,
                "label": profile.label,
                "reasoning_effort": profile.reasoning_effort,
                "ollama_think": profile.ollama_think,
                "num_predict_multiplier": profile.num_predict_multiplier,
                "force_schema_grounding": profile.force_schema_grounding,
            }
            for profile in list_profiles()
        ]
    }


@app.get("/health")
async def health() -> dict:
    try:
        ollama = await engine.client.version()
        return {
            "status": "ok",
            "engine_version": __version__,
            "model": settings.ollama_model,
            "prewarm_on_start": settings.prewarm_on_start,
            "keep_alive": settings.keep_alive,
            "ollama": ollama,
            "research": {
                "provider": settings.research_provider,
                "configured": bool(settings.searxng_base_url),
                "fetch_enabled": settings.research_fetch_enabled,
            },
            "final_audit": {
                "enabled": settings.final_audit_enabled,
                "fact_audit": settings.final_audit_fact_enabled,
                "language_audit": settings.final_audit_language_enabled,
                "reverse_research": settings.final_audit_reverse_research_enabled,
            },
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama unavailable: {exc}",
        ) from exc


@app.post(
    "/v1/warmup",
    dependencies=[Depends(require_api_key)],
)
async def warmup() -> dict:
    result = await engine.client.warmup(
        model=settings.ollama_model,
        keep_alive=settings.keep_alive,
        num_ctx=settings.num_ctx,
    )
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "keep_alive": settings.keep_alive,
        "load_duration_seconds": (
            result.load_duration_ns / 1_000_000_000
            if result.load_duration_ns
            else 0.0
        ),
        "total_duration_seconds": (
            result.total_duration_ns / 1_000_000_000
            if result.total_duration_ns
            else 0.0
        ),
    }


@app.post(
    "/v1/schema/preview",
    dependencies=[Depends(require_api_key)],
)
async def preview_schema(request: GenerationRequest) -> dict:
    return build_conversation_schema(request)


@app.post(
    "/v1/conversations/generate",
    response_model=GenerationResponse,
    dependencies=[Depends(require_api_key)],
)
async def generate_conversation(
    request: GenerationRequest,
) -> GenerationResponse:
    try:
        return await engine.generate(request)
    except ResearchUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": str(exc), "type": "research_unavailable"},
        ) from exc
    except ResearchError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": str(exc), "type": "research_failed"},
        ) from exc
    except GenerationFailed as exc:
        detail = {
            "message": str(exc),
            "violations": [
                item.model_dump()
                for item in exc.violations
            ],
        }
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        ) from exc
