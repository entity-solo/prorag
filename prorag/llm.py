"""
LLM adapter — single call interface, provider-agnostic.

Default: Groq (fast, free tier available).
Swap to OpenAI, Anthropic, or local Ollama by setting PRORAG_LLM_PROVIDER.
"""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_CACHE_LOCK = threading.Lock()
_CACHE_SENTINEL = object()


def call_llm(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1024,
    system: str = "",
) -> str:
    """Single synchronous LLM call. Returns the assistant text."""
    provider = os.getenv("PRORAG_LLM_PROVIDER", "groq").lower()
    cache_key = _cache_key(provider, model, max_tokens, system, prompt)
    cached = _cache_get(cache_key)
    if cached is not _CACHE_SENTINEL:
        return str(cached)

    if provider == "groq":
        response = _groq(prompt, model=model, max_tokens=max_tokens, system=system)
    elif provider == "openai":
        response = _openai(prompt, model=model, max_tokens=max_tokens, system=system)
    elif provider == "ollama":
        response = _ollama(prompt, model=model, max_tokens=max_tokens)
    elif provider == "anthropic":
        response = _anthropic(prompt, model=model, max_tokens=max_tokens, system=system)
    else:
        raise ValueError(f"Unknown PRORAG_LLM_PROVIDER: {provider!r}")

    _cache_set(cache_key, response)
    return response


def _cache_enabled() -> bool:
    return os.getenv("PRORAG_LLM_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_path() -> Path:
    return Path(os.getenv("PRORAG_LLM_CACHE_PATH", ".prorag_cache/llm_cache.json"))


def _cache_key(provider: str, model: str, max_tokens: int, system: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "prompt": prompt,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | object:
    if not _cache_enabled():
        return _CACHE_SENTINEL
    path = _cache_path()
    if not path.exists():
        return _CACHE_SENTINEL
    with _CACHE_LOCK:
        try:
            with open(path, encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return _CACHE_SENTINEL
    return cache.get(key, _CACHE_SENTINEL)


def _cache_set(key: str, value: str) -> None:
    if not _cache_enabled():
        return
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        try:
            with open(path, encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, json.JSONDecodeError):
            cache = {}
        cache[key] = value
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            json.dump(cache, handle, ensure_ascii=False)
            temp_path = Path(handle.name)
        temp_path.replace(path)


# ── providers ─────────────────────────────────────────────────────────────────

def _groq(prompt, model, max_tokens, system) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")
    import time

    api_key = os.environ.get("GROQ_API_KEY") or _missing("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    max_retries = 8
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
            return resp.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(term in err_str for term in ("429", "rate limit", "limit_reached", "tp_limit_reached", "rate_limit_exceeded"))
            is_conn_error = any(term in err_str for term in ("connection error", "connecterror", "10051", "unreachable network", "timeout"))
            
            if is_rate_limit or is_conn_error:
                wait_time = (2 ** attempt) * 2
                error_type = "Rate Limit" if is_rate_limit else "Connection Error"
                print(f"[{error_type}] Groq call failed. Waiting {wait_time}s before retry... (Error: {e})")
                time.sleep(wait_time)
            else:
                raise e

    raise RuntimeError("Failed to call Groq API after maximum retries due to errors.")


def _openai(prompt, model, max_tokens, system) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")
    import time

    api_key = os.environ.get("OPENAI_API_KEY") or _missing("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    max_retries = 8
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
            return resp.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(
                term in err_str
                for term in ("429", "rate limit", "rate_limit", "too many requests")
            )
            is_conn_error = any(
                term in err_str
                for term in ("connection error", "connecterror", "timeout", "unreachable")
            )
            if is_rate_limit or is_conn_error:
                wait_time = (2 ** attempt) * 2
                error_type = "Rate Limit" if is_rate_limit else "Connection Error"
                print(f"[{error_type}] OpenAI-compatible call failed. Waiting {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("Failed OpenAI-compatible API call after maximum retries.")


def _ollama(prompt, model, max_tokens) -> str:
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resp = requests.post(
        f"{base}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _anthropic(prompt, model, max_tokens, system) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or _missing("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    return msg.content[0].text if msg.content else ""


def _missing(var: str) -> None:
    raise EnvironmentError(f"Set the {var} environment variable.")
