#!/usr/bin/env python3
"""Teacher-driven synthetic data generation pipeline.

Generates instruction/response pairs using a teacher model (local MLX or any
OpenAI-compatible endpoint), applies quality filtering and deduplication, then
saves to a parquet file ready for fine-tuning with main.py.

Usage:
    uv run python generate_data.py --config configs/mentor_style_local.yaml
    uv run python generate_data.py --config configs/mentor_style_openai.yaml
    uv run python generate_data.py --config configs/mentor_style_local.yaml --dry-run
"""
from dotenv import load_dotenv

load_dotenv()

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from finetune.config import load_config, ExperimentConfig
from finetune.generation.prompt_builder import load_seeds, build_instructions
from finetune.generation.examples import load_examples
from finetune.generation.quality import GeneratedPair, filter_pairs, run_dedup
from finetune.data.parquet_store import already_generated_seeds, append_pairs
from finetune.teacher.base import TeacherBackend


def build_teacher(
    cfg: ExperimentConfig,
    examples: list[dict[str, str]],
) -> TeacherBackend:
    """Factory: return the correct TeacherBackend based on config."""
    if cfg.teacher.backend == "local_mlx":
        from finetune.teacher.local_mlx import LocalMLXTeacher

        return LocalMLXTeacher(cfg.teacher, examples=examples)
    if cfg.teacher.backend == "openai_api":
        from finetune.teacher.openai_api import OpenAICompatibleTeacher

        return OpenAICompatibleTeacher(cfg.teacher, examples=examples)
    raise ValueError(
        f"Unknown teacher backend: {cfg.teacher.backend!r}. "
        "Choose from: local_mlx, openai_api"
    )


def run_generation(cfg: ExperimentConfig, dry_run: bool = False) -> None:
    """Run the full data-generation pipeline.

    Steps:
        1. Load few-shot examples (optional).
        2. Load and expand seed prompts into instructions.
        3. Skip already-generated seeds when ``cfg.generation.resume`` is True.
        4. Generate responses with the teacher in batches.
        5. Apply quality filters and deduplication after each batch.
        6. Append surviving pairs to the parquet file atomically.

    Args:
        cfg: Fully loaded experiment configuration.
        dry_run: When True, print the first few instructions and return
                 without calling the teacher model.
    """
    output_path = Path(cfg.generation.output_dir) / cfg.generation.output_file

    # 1. Load few-shot examples (optional)
    rng = random.Random(42)
    examples: list[dict[str, str]] = []
    if cfg.teacher.examples_file:
        examples = load_examples(
            cfg.teacher.examples_file,
            max_examples=cfg.teacher.max_examples,
            rng=rng,
        )
        print(
            f"Using {len(examples)} few-shot examples from {cfg.teacher.examples_file}"
        )

    # 2. Load seed prompts
    all_seeds = load_seeds(cfg.seeds.files)
    if not all_seeds:
        raise RuntimeError("No seed prompts loaded. Check seeds.files in your config.")
    print(f"Loaded {len(all_seeds)} seed prompts from {cfg.seeds.files}")

    # 3. Resume: skip seeds already present in the parquet
    seeds_to_run = all_seeds
    if cfg.generation.resume:
        done = already_generated_seeds(output_path)
        if done:
            seeds_to_run = [s for s in all_seeds if s not in done]
            print(
                f"Resume: skipping {len(done)} already-generated seeds, "
                f"{len(seeds_to_run)} remaining."
            )

    if not seeds_to_run:
        print("All seeds already generated. Set resume: false in config to regenerate.")
        return

    # 4. Expand seeds into (instruction, source_seed) pairs
    instruction_pairs = build_instructions(
        seeds_to_run,
        cfg.seeds.samples_per_seed,
        cfg.seeds.augment,
        rng=rng,
    )
    print(f"Will generate {len(instruction_pairs)} instruction/response pairs.")

    if dry_run:
        print("\n[dry-run] First 5 instructions:")
        for inst, seed in instruction_pairs[:5]:
            print(f"  seed={seed!r}")
            print(f"  instruction={inst!r}\n")
        if examples:
            print(f"[dry-run] First few-shot example:")
            print(f"  Q: {examples[0]['instruction']}")
            print(f"  A: {examples[0]['output'][:120]}...")
        return

    # 5. Generate with teacher
    teacher = build_teacher(cfg, examples)
    batch_size = cfg.generation.batch_size

    try:
        for i in tqdm(
            range(0, len(instruction_pairs), batch_size),
            desc="Generating",
            unit="batch",
        ):
            batch = instruction_pairs[i : i + batch_size]
            instructions = [inst for inst, _ in batch]
            source_seeds = [seed for _, seed in batch]

            try:
                outputs = teacher.generate_batch(
                    instructions, cfg.teacher.system_prompt
                )
            except Exception as e:
                print(f"\nError in batch {i}: {e}")
                outputs = [""] * len(instructions)

            pairs = [
                GeneratedPair(
                    instruction=inst,
                    output=out,
                    source_seed=src,
                    teacher_model=cfg.teacher.model,
                )
                for inst, out, src in zip(instructions, outputs, source_seeds)
            ]

            # 6. Filter + dedup + save after each batch (crash-safe)
            pairs = filter_pairs(pairs, cfg.quality)
            pairs = run_dedup(pairs, cfg.quality)
            append_pairs(pairs, output_path)

    finally:
        teacher.close()

    print(f"\nGeneration complete. Data saved to: {output_path}")
    print(
        f"Next step: uv run python main.py train --data-dir {cfg.generation.output_dir}"
    )


def main() -> None:
    """Parse CLI arguments and invoke the generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using a teacher model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config (e.g. configs/mentor_style_local.yaml)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Path to base YAML config with shared defaults",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview instructions and examples without calling the teacher model",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base_config)
    print(f"Experiment : {cfg.experiment_name}")
    print(f"Teacher    : {cfg.teacher.backend} / {cfg.teacher.model}")
    print(f"Examples   : {cfg.teacher.examples_file or 'none'}")
    print(f"Student    : {cfg.student.model}")
    print(f"Output     : {cfg.generation.output_dir}/{cfg.generation.output_file}\n")

    run_generation(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
