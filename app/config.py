from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AY_",
        extra="ignore",
    )

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gpt-oss:20b"
    api_key: str = ""

    num_ctx: int = 4096
    keep_alive: str = "30m"

    max_repair_attempts: int = 2
    request_timeout_seconds: float = 180.0
    ground_schema_in_prompt: bool = False

    # Load the conversation model when the API service starts so the first
    # real user request does not pay the ~10 s cold-load penalty.
    prewarm_on_start: bool = True

    # Optional background research. The public request only exposes one switch;
    # these operational details stay server-side.
    research_provider: str = "searxng"
    searxng_base_url: str = ""
    research_timeout_seconds: float = 10.0
    research_max_queries: int = 2
    research_max_followup_queries: int = 1
    research_max_sources: int = 5
    research_safe_search: int = 1

    # Optional direct source retrieval after SearXNG discovery. Search-result
    # snippets are often too short for technical fact checking, so the best
    # exact-identity sources can be fetched and reduced to relevant excerpts.
    research_fetch_enabled: bool = True
    research_fetch_max_sources: int = 3
    research_fetch_timeout_seconds: float = 8.0
    research_fetch_max_bytes: int = 5_000_000
    research_fetch_max_chars: int = 2400
    research_fetch_max_redirects: int = 3
    research_fetch_pdf_max_pages: int = 40

    # Final self-audit before returning JSON. Fact audit may use reverse web research
    # only when the public request has features.research=true. Language/coherence
    # audit is local/model-only and is not allowed to alter factual anchors.
    final_audit_enabled: bool = True
    final_audit_fact_enabled: bool = True
    final_audit_language_enabled: bool = True
    final_audit_reverse_research_enabled: bool = True
    final_audit_max_reverse_queries: int = 3
    final_audit_reverse_max_sources: int = 3
    final_audit_num_predict: int = 800
    # Audit classification/rewrite tasks are narrow and deterministic; disable
    # model thinking by default to reduce latency independently of generation.
    final_audit_think: bool | str = "low"


settings = Settings()
