"""Data quality filtering and deduplication for generated pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from finetune.config import QualityConfig

# Patterns that indicate the teacher refused or couldn't complete the request.
# Matching any of these causes the pair to be dropped during quality filtering.
_REFUSAL_PATTERNS = [
    re.compile(r"i('m| am) (sorry|unable|not able)", re.IGNORECASE),
    re.compile(r"as an ai( language model)?", re.IGNORECASE),
    re.compile(r"i can'?t (assist|help) with that", re.IGNORECASE),
    re.compile(r"i (won'?t|cannot) (generate|provide|create)", re.IGNORECASE),
]


@dataclass
class GeneratedPair:
    """A single instruction/output pair produced by the teacher.

    Attributes:
        instruction: The user-facing instruction sent to the teacher.
        output: The teacher's generated response.
        source_seed: The original seed prompt this instruction was derived from.
        teacher_model: Model ID of the teacher that produced this output.
    """

    instruction: str
    output: str
    source_seed: str = ""
    teacher_model: str = ""


def _approx_tokens(text: str) -> int:
    """Whitespace-based token count approximation (no tokenizer needed)."""
    return len(text.split())


def _is_refusal(text: str) -> bool:
    """Return True if the text matches any known refusal pattern."""
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


def filter_pairs(pairs: list[GeneratedPair], cfg: QualityConfig) -> list[GeneratedPair]:
    """Apply quality filters: length bounds, empty-output guard, refusal detection."""
    kept: list[GeneratedPair] = []
    rejected = {
        "empty": 0,
        "short_output": 0,
        "long_output": 0,
        "short_instruction": 0,
        "long_instruction": 0,
        "refusal": 0,
    }

    for pair in pairs:
        if not pair.output.strip():
            rejected["empty"] += 1
            continue
        out_tokens = _approx_tokens(pair.output)
        inst_tokens = _approx_tokens(pair.instruction)
        if out_tokens < cfg.min_output_tokens:
            rejected["short_output"] += 1
            continue
        if out_tokens > cfg.max_output_tokens:
            rejected["long_output"] += 1
            continue
        if inst_tokens < cfg.min_instruction_tokens:
            rejected["short_instruction"] += 1
            continue
        if inst_tokens > cfg.max_instruction_tokens:
            rejected["long_instruction"] += 1
            continue
        if cfg.filter_refusals and _is_refusal(pair.output):
            rejected["refusal"] += 1
            continue
        kept.append(pair)

    total_rejected = sum(rejected.values())
    if total_rejected:
        print(f"Quality filter: removed {total_rejected} pairs — {rejected}")

    return kept


def deduplicate_exact(pairs: list[GeneratedPair]) -> list[GeneratedPair]:
    """Remove exact-duplicate outputs (case-insensitive, stripped)."""
    seen: set[str] = set()
    unique: list[GeneratedPair] = []
    for pair in pairs:
        key = pair.output.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    removed = len(pairs) - len(unique)
    if removed:
        print(f"Exact dedup: removed {removed} duplicate outputs.")
    return unique


def deduplicate_semantic(
    pairs: list[GeneratedPair],
    threshold: float = 0.92,
) -> list[GeneratedPair]:
    """Remove near-duplicate outputs via cosine similarity of sentence embeddings.

    Requires: uv add sentence-transformers
    O(n²) — only suitable for datasets < ~5k pairs.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print(
            "Warning: sentence-transformers is not installed — semantic dedup disabled, "
            "falling back to exact dedup. Install with: uv add sentence-transformers"
        )
        return deduplicate_exact(pairs)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        [p.output for p in pairs],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    kept_indices: list[int] = []
    for i, emb in enumerate(embeddings):
        is_dup = any(
            float(emb @ embeddings[j]) > threshold for j in kept_indices
        )
        if not is_dup:
            kept_indices.append(i)

    result = [pairs[i] for i in kept_indices]
    print(f"Semantic dedup: removed {len(pairs) - len(result)} near-duplicates (threshold={threshold}).")
    return result


def run_dedup(pairs: list[GeneratedPair], cfg: QualityConfig) -> list[GeneratedPair]:
    """Dispatch to the configured deduplication strategy."""
    if cfg.dedup_strategy == "none":
        return pairs
    if cfg.dedup_strategy == "exact":
        return deduplicate_exact(pairs)
    if cfg.dedup_strategy == "semantic":
        pairs = deduplicate_exact(pairs)
        return deduplicate_semantic(pairs, threshold=cfg.semantic_threshold)
    raise ValueError(f"Unknown dedup_strategy: {cfg.dedup_strategy!r}")
