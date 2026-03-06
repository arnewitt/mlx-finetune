"""Load and validate experiment configuration from YAML files.

Configuration is stored as plain YAML.  Each experiment YAML merges on top of
``configs/base.yaml``; only the keys that differ from the base need to be
specified.  All sections are typed dataclasses, so IDE completion and runtime
validation are available out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VALID_DEDUP_STRATEGIES = ("none", "exact", "semantic")

_KNOWN_SECTIONS = {"experiment_name", "teacher", "seeds", "generation", "quality", "student", "mlflow"}


@dataclass
class TeacherConfig:
    """Configuration for the teacher model used during data generation.

    Attributes:
        backend: Which backend to use — ``"local_mlx"`` or ``"openai_api"``.
        model: Model ID (HuggingFace path for local_mlx, model name for API).
        max_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature; higher = more diverse outputs.
        system_prompt: System message injected before every conversation.
        base_url: API base URL for openai_api (None = OpenAI default).
        api_key: API key for openai_api; falls back to ``OPENAI_API_KEY`` env var.
        examples_file: Path to a JSONL file with few-shot example pairs.
        max_examples: Number of examples to sample per generation call.
    """

    backend: str = "local_mlx"       # "local_mlx" | "openai_api"
    model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    max_tokens: int = 512
    temperature: float = 0.8
    system_prompt: str = "You are a helpful assistant."
    # openai_api only: full base URL including /v1 (None = OpenAI default)
    base_url: str | None = None
    # openai_api only: falls back to OPENAI_API_KEY env var if not set
    api_key: str | None = None
    # path to a JSONL file with {"instruction": ..., "output": ...} few-shot examples
    examples_file: str | None = None
    # how many examples to sample from the file per generation call
    max_examples: int = 5
    # extra fields forwarded verbatim to chat.completions.create (openai_api only)
    # e.g. {"chat_template_kwargs": {"thinking": false}} to disable reasoning for Kimi-K2.5
    extra_body: dict | None = None


@dataclass
class SeedConfig:
    """Configuration for seed prompt loading and augmentation.

    Attributes:
        files: List of text files, one seed prompt per non-comment line.
        samples_per_seed: How many instruction variants to generate per seed.
        augment: When True, paraphrase templates are applied to each seed.
    """

    files: list[str] = field(default_factory=lambda: ["seeds/general_knowledge.txt"])
    samples_per_seed: int = 3
    augment: bool = False


@dataclass
class GenerationConfig:
    """Configuration for the data generation output and batching.

    Attributes:
        output_dir: Directory where the parquet file is written.
        output_file: Parquet filename within ``output_dir``.
        batch_size: Number of prompts sent to the teacher per batch.
        resume: When True, seeds already in the parquet file are skipped.
    """

    output_dir: str = "data"
    output_file: str = "generated.parquet"
    batch_size: int = 8
    resume: bool = True


@dataclass
class QualityConfig:
    """Configuration for quality filtering and deduplication.

    Attributes:
        min_output_tokens: Minimum word-count for generated outputs.
        max_output_tokens: Maximum word-count for generated outputs.
        min_instruction_tokens: Minimum word-count for instructions.
        max_instruction_tokens: Maximum word-count for instructions.
        dedup_strategy: One of ``"none"``, ``"exact"``, or ``"semantic"``.
        semantic_threshold: Cosine similarity cutoff for semantic dedup.
        filter_refusals: When True, drop outputs matching refusal patterns.
    """

    min_output_tokens: int = 40
    max_output_tokens: int = 600
    min_instruction_tokens: int = 2
    max_instruction_tokens: int = 200
    dedup_strategy: str = "exact"
    semantic_threshold: float = 0.92
    filter_refusals: bool = True


@dataclass
class StudentConfig:
    """Configuration for the student model fine-tuning run.

    Attributes:
        model: HuggingFace model ID or local path for the student.
        iters: Number of LoRA training iterations.
        batch_size: Training batch size (auto-reduced for small datasets).
        num_layers: Number of transformer layers to apply LoRA to (-1 = all).
        learning_rate: AdamW learning rate.
        valid_split: Fraction of data held out for validation (0 = disabled).
        grad_checkpoint: Enable gradient checkpointing to reduce memory usage.
        max_seq_length: Maximum sequence length (longer examples are truncated).
    """

    model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    iters: int = 200
    batch_size: int = 4
    num_layers: int = 16
    learning_rate: float = 1e-5
    valid_split: float = 0.1
    grad_checkpoint: bool = False
    max_seq_length: int = 2048
    steps_per_eval: int = 50


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking.

    Attributes:
        tracking_uri: URI of the MLflow tracking server.
        experiment: MLflow experiment name under which runs are logged.
    """

    tracking_uri: str = "http://localhost:5002"
    experiment: str = "teacher-student"


@dataclass
class ExperimentConfig:
    """Top-level configuration object representing one experiment.

    Aggregates all section configs and the experiment name.  Produced by
    :func:`load_config` after merging base.yaml with an experiment YAML.
    """

    experiment_name: str = "unnamed"
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; all other values are replaced.

    Args:
        base: The base dictionary (e.g. parsed base.yaml).
        override: The overriding dictionary (e.g. parsed experiment YAML).

    Returns:
        A new merged dictionary without modifying either input.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(
    config_path: str | Path,
    base_path: str | Path = "configs/base.yaml",
) -> ExperimentConfig:
    """Load a YAML config file, merging it on top of ``base.yaml`` defaults.

    Unknown top-level section keys emit a warning rather than raising, so
    typos in config files are surfaced immediately without aborting.

    Args:
        config_path: Path to the experiment-specific YAML file.
        base_path: Path to the shared base YAML (default: ``configs/base.yaml``).

    Returns:
        A fully populated :class:`ExperimentConfig`.

    Raises:
        ValueError: If ``quality.dedup_strategy`` is not one of the valid values.
        FileNotFoundError: If ``config_path`` does not exist.
    """
    base_path = Path(base_path)
    config_path = Path(config_path)

    raw: dict[str, Any] = {}
    if base_path.exists():
        with open(base_path) as f:
            raw = yaml.safe_load(f) or {}

    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    merged = _deep_merge(raw, override)

    # Warn about unknown top-level keys (catches typos like "teachers:" or "mlflows:")
    for key in merged:
        if key not in _KNOWN_SECTIONS:
            print(f"Warning: unknown config section {key!r} — did you make a typo?")

    cfg = ExperimentConfig(experiment_name=merged.get("experiment_name", "unnamed"))
    section_map: list[tuple[str, type]] = [
        ("teacher", TeacherConfig),
        ("seeds", SeedConfig),
        ("generation", GenerationConfig),
        ("quality", QualityConfig),
        ("student", StudentConfig),
        ("mlflow", MLflowConfig),
    ]
    for section, cls in section_map:
        if section in merged:
            setattr(cfg, section, cls(**merged[section]))

    # Validate dedup_strategy after the quality section is set
    if cfg.quality.dedup_strategy not in _VALID_DEDUP_STRATEGIES:
        raise ValueError(
            f"Invalid dedup_strategy {cfg.quality.dedup_strategy!r}. "
            f"Must be one of: {_VALID_DEDUP_STRATEGIES}"
        )

    return cfg
