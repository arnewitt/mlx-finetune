"""Seed loading and instruction construction for the teacher."""

from __future__ import annotations

import random
from pathlib import Path

# Paraphrase templates create surface variation from a single seed prompt
_AUGMENT_TEMPLATES = [
    "{seed}",
    "Can you explain: {seed}",
    "In simple terms, {seed}",
    "I'd like to understand {seed}",
    "Help me understand this: {seed}",
    "What's the best way to think about {seed}?",
]


def load_seeds(seed_files: list[str | Path]) -> list[str]:
    """Load all seed prompts from text files.

    Skips blank lines and lines starting with '#'.
    """
    seeds: list[str] = []
    for path in seed_files:
        p = Path(path)
        if not p.exists():
            print(f"Warning: seed file not found: {p}")
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    seeds.append(line)
    return seeds


def build_instructions(
    seeds: list[str],
    samples_per_seed: int,
    augment: bool,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    """Expand seeds into (instruction, source_seed) pairs.

    When augment=True, different paraphrase templates are applied to each
    sample, creating surface variation while preserving semantic intent.
    Teacher temperature creates output diversity even without augmentation.

    Returns a list of (instruction, source_seed) tuples.
    """
    if rng is None:
        rng = random.Random(42)

    pairs: list[tuple[str, str]] = []
    for seed in seeds:
        if augment and samples_per_seed > 1:
            templates = rng.sample(
                _AUGMENT_TEMPLATES,
                min(samples_per_seed, len(_AUGMENT_TEMPLATES)),
            )
            while len(templates) < samples_per_seed:
                templates.append(rng.choice(_AUGMENT_TEMPLATES))
            for tmpl in templates[:samples_per_seed]:
                pairs.append((tmpl.format(seed=seed), seed))
        else:
            for _ in range(samples_per_seed):
                pairs.append((seed, seed))

    return pairs
