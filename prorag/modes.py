"""
Cost and quality presets for LLM-backed ProRAG operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QualityMode = Literal["cheap", "balanced", "quality"]


@dataclass(frozen=True)
class ModeConfig:
    entity_max_tokens: int
    extraction_max_tokens: int
    answer_max_tokens: int
    entity_max_retries: int
    entity_expand_sentences: int


_MODE_CONFIGS: dict[QualityMode, ModeConfig] = {
    "cheap": ModeConfig(
        entity_max_tokens=1024,
        extraction_max_tokens=1536,
        answer_max_tokens=256,
        entity_max_retries=1,
        entity_expand_sentences=3,
    ),
    "balanced": ModeConfig(
        entity_max_tokens=2048,
        extraction_max_tokens=3072,
        answer_max_tokens=512,
        entity_max_retries=2,
        entity_expand_sentences=4,
    ),
    "quality": ModeConfig(
        entity_max_tokens=4096,
        extraction_max_tokens=4096,
        answer_max_tokens=1024,
        entity_max_retries=3,
        entity_expand_sentences=6,
    ),
}


def get_mode_config(mode: QualityMode | str = "balanced") -> ModeConfig:
    try:
        return _MODE_CONFIGS[mode]  # type: ignore[index]
    except KeyError:
        allowed = ", ".join(sorted(_MODE_CONFIGS))
        raise ValueError(f"Unknown quality mode: {mode!r}. Expected one of: {allowed}.")


def available_modes() -> tuple[str, ...]:
    return tuple(_MODE_CONFIGS)
