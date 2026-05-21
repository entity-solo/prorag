"""
LLM adapter — single call interface, provider-agnostic.

Default: Groq (fast, free tier available).
Swap to OpenAI, Anthropic, or local Ollama by setting PRORAG_LLM_PROVIDER.
"""

import os
from typing import Any


def call_llm(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1024,
    system: str = "",
) -> str:
    """Single synchronous LLM call. Returns the assistant text."""
    provider = os.getenv("PRORAG_LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        return _groq(prompt, model=model, max_tokens=max_tokens, system=system)
    elif provider == "openai":
        return _openai(prompt, model=model, max_tokens=max_tokens, system=system)
    elif provider == "ollama":
        return _ollama(prompt, model=model, max_tokens=max_tokens)
    elif provider == "anthropic":
        return _anthropic(prompt, model=model, max_tokens=max_tokens, system=system)
    else:
        raise ValueError(f"Unknown PRORAG_LLM_PROVIDER: {provider!r}")


# ── providers ─────────────────────────────────────────────────────────────────

def _groq(prompt, model, max_tokens, system) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    api_key = os.environ.get("GROQ_API_KEY") or _missing("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


def _openai(prompt, model, max_tokens, system) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY") or _missing("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


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
