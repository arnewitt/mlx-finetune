"""Load and sample few-shot examples from a JSONL file.

Format — one JSON object per line:
    {"instruction": "...", "output": "..."}

The examples are injected as user/assistant turns in the teacher's
message history before the new instruction, giving the model concrete
demonstrations of the desired style, tone, and length.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_examples(
    examples_file: str | Path,
    max_examples: int = 5,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Load up to `max_examples` pairs from a JSONL file.

    If the file has more examples than `max_examples`, a random subset
    is sampled so the teacher sees variety across generation batches.

    Each returned dict has keys: "instruction", "output".
    """
    path = Path(examples_file)
    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {path}")

    pairs: list[dict[str, str]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e

            if "instruction" not in obj or "output" not in obj:
                raise ValueError(
                    f"Line {lineno} of {path} must have 'instruction' and 'output' keys. "
                    f"Got: {list(obj.keys())}"
                )
            instruction = obj["instruction"]
            output = obj["output"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f"Line {lineno} of {path}: 'instruction' must be a non-empty string."
                )
            if not isinstance(output, str) or not output.strip():
                raise ValueError(
                    f"Line {lineno} of {path}: 'output' must be a non-empty string."
                )
            pairs.append({"instruction": instruction, "output": output})

    if not pairs:
        raise ValueError(f"No examples found in {path}")

    if len(pairs) <= max_examples:
        return pairs

    if rng is None:
        rng = random.Random(42)
    return rng.sample(pairs, max_examples)
