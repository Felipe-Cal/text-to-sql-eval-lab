"""Inference backend configuration.

Centralises all LiteLLM call-site settings so swapping from OpenAI to a
local vLLM server is a pure .env change — no Python edits required.

Usage:
    from src.inference.backend import get_completion_backend, build_completion_kwargs, supports_json_schema

    backend = get_completion_backend()
    kwargs = build_completion_kwargs(backend, model=model, temperature=0, messages=msgs)
    response = litellm.completion(**kwargs)

vLLM activation — set all three env vars:
    VLLM_API_BASE=http://localhost:8000/v1
    VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
    VLLM_API_KEY=token-abc123          # omit or leave blank for no-auth servers
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class InferenceBackend:
    """Resolved inference backend — immutable snapshot of env config."""

    model_name: str
    api_base: str | None  # None → LiteLLM uses its built-in routing (OpenAI, Anthropic, …)
    api_key: str | None
    is_vllm: bool


def get_completion_backend(model: str | None = None) -> InferenceBackend:
    """
    Return the active completion backend.

    Priority:
      1. vLLM  — when model matches VLLM_MODEL and VLLM_API_BASE is set.
      2. Default — model param or DEFAULT_MODEL env var (LiteLLM routes to OpenAI, Anthropic, etc.).
    """
    vllm_base = os.getenv("VLLM_API_BASE", "").strip()
    vllm_model = os.getenv("VLLM_MODEL", "").strip()

    # Determine which model is being requested
    requested_model = model or os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")

    # If it's specifically the vLLM model and we have a base URL, use vLLM
    if vllm_base and vllm_model and requested_model == vllm_model:
        return InferenceBackend(
            model_name=vllm_model,
            api_base=vllm_base,
            api_key=os.getenv("VLLM_API_KEY") or None,
            is_vllm=True,
        )

    return InferenceBackend(
        model_name=requested_model,
        api_base=None,
        api_key=None,
        is_vllm=False,
    )


def get_embedding_backend() -> InferenceBackend:
    """
    Return the active embedding backend.

    Priority:
      1. vLLM embedding model — when VLLM_API_BASE and VLLM_EMBEDDING_MODEL are both set.
      2. EMBEDDING_MODEL env var (OpenAI embeddings by default).
    """
    vllm_base = os.getenv("VLLM_API_BASE", "").strip()
    vllm_embed = os.getenv("VLLM_EMBEDDING_MODEL", "").strip()

    if vllm_base and vllm_embed:
        return InferenceBackend(
            model_name=vllm_embed,
            api_base=vllm_base,
            api_key=os.getenv("VLLM_API_KEY") or None,
            is_vllm=True,
        )

    return InferenceBackend(
        model_name=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_base=None,
        api_key=None,
        is_vllm=False,
    )


def build_completion_kwargs(
    backend: InferenceBackend,
    model: str | None = None,
    **overrides,
) -> dict:
    """
    Build a kwargs dict ready to **unpack into litellm.completion() or acompletion().

    ``model`` overrides backend.model_name — used when the router picks a specific
    model (e.g. HARD_MODEL) that should still be served through the vLLM endpoint.

    vLLM-specific defaults injected automatically:
      - api_base        — points LiteLLM at the local vLLM server
      - api_key         — passed through if set (some vLLM deployments require a token)
      - extra_body      — disables beam search to enforce greedy decoding at temperature=0
    """
    effective_model = model or backend.model_name

    # vLLM exposes an OpenAI-compatible API.  LiteLLM requires the "openai/"
    # prefix to route to a custom base URL correctly.
    if backend.is_vllm and not effective_model.startswith("openai/"):
        effective_model = f"openai/{effective_model}"

    kwargs: dict = {"model": effective_model}

    if backend.api_base:
        kwargs["api_base"] = backend.api_base
    if backend.api_key:
        kwargs["api_key"] = backend.api_key

    # vLLM: beam search conflicts with temperature=0 greedy decoding
    if backend.is_vllm:
        kwargs["extra_body"] = {"use_beam_search": False}

    kwargs.update(overrides)
    return kwargs


def supports_json_schema(backend: InferenceBackend, model: str | None = None) -> bool:
    """
    Return True if this model/backend supports OpenAI-style JSON schema constrained
    decoding via ``response_format={"type": "json_schema", ...}``.

    vLLM always supports it (OpenAI-compatible endpoint with guided decoding).
    OpenAI gpt-4o-* and gpt-4-turbo support it natively.
    Anthropic, Mistral, and older models fall back to regex/Instructor extraction.
    """
    if backend.is_vllm:
        return True
    effective_model = (model or backend.model_name).lower()
    return any(
        prefix in effective_model
        for prefix in ("gpt-4o", "gpt-4-turbo", "o1-", "o3-")
    )
