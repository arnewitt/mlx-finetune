#!/usr/bin/env python3
"""Fine-tune instruct models using Apple Silicon MLX with LoRA."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

from finetune.config import load_config

from mlx_lm import generate, load
import mlx_lm.lora as _mlx_lora
from mlx_lm.lora import (
    CONFIG_DEFAULTS as LORA_DEFAULTS,
    build_parser as build_lora_parser,
    run as lora_run,
)
from mlx_lm.tuner.callbacks import TrainingCallback


DATA_DIR = Path("data")
CACHE_DIR = Path(".cache")
DEFAULT_MLFLOW_URI = "http://localhost:5002"


class MLflowCallback(TrainingCallback):
    """Callback to log training metrics to MLflow."""

    def on_train_loss_report(self, train_info: dict) -> None:
        """Log training loss to MLflow."""
        step = train_info.get("iteration", 0)
        mlflow.log_metric("train_loss", train_info.get("train_loss", 0), step=step)
        if "learning_rate" in train_info:
            mlflow.log_metric("learning_rate", train_info["learning_rate"], step=step)
        if "tokens_per_second" in train_info:
            mlflow.log_metric(
                "tokens_per_second", train_info["tokens_per_second"], step=step
            )
        if "iterations_per_second" in train_info:
            mlflow.log_metric(
                "iterations_per_second", train_info["iterations_per_second"], step=step
            )
        if "trained_tokens" in train_info:
            mlflow.log_metric("trained_tokens", train_info["trained_tokens"], step=step)
        if "peak_memory" in train_info:
            mlflow.log_metric("peak_memory_gb", train_info["peak_memory"], step=step)

    def on_val_loss_report(self, val_info: dict) -> None:
        """Log validation loss to MLflow."""
        step = val_info.get("iteration", 0)
        mlflow.log_metric("val_loss", val_info.get("val_loss", 0), step=step)


def load_parquet_dataset(data_dir: Path) -> pd.DataFrame:
    """Load all parquet files from the data directory."""
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(
        f"Found {len(parquet_files)} parquet file(s): {[f.name for f in parquet_files]}"
    )
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def detect_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    """Detect instruction, output, and optional system columns."""
    columns = set(df.columns.str.lower())
    df_columns = {c.lower(): c for c in df.columns}

    instruction_candidates = ["instruction", "prompt", "input", "question", "user"]
    instruction_col = None
    for candidate in instruction_candidates:
        if candidate in columns:
            instruction_col = df_columns[candidate]
            break

    output_candidates = ["output", "response", "completion", "answer", "assistant"]
    output_col = None
    for candidate in output_candidates:
        if candidate in columns:
            output_col = df_columns[candidate]
            break

    system_col = df_columns.get("system")

    if not instruction_col or not output_col:
        raise ValueError(
            f"Could not detect required columns. Found: {list(df.columns)}. "
            f"Expected instruction column (one of: {instruction_candidates}) "
            f"and output column (one of: {output_candidates})"
        )

    print(
        f"Detected columns - instruction: '{instruction_col}', output: '{output_col}'"
        + (f", system: '{system_col}'" if system_col else "")
    )

    return instruction_col, output_col, system_col


def convert_to_chat_format(
    df: pd.DataFrame,
    instruction_col: str,
    output_col: str,
    system_col: str | None,
) -> list[dict]:
    """Convert dataframe rows to chat message format."""
    records = []
    for _, row in df.iterrows():
        messages = []

        if system_col and pd.notna(row.get(system_col)):
            messages.append({"role": "system", "content": str(row[system_col])})

        messages.append({"role": "user", "content": str(row[instruction_col])})
        messages.append({"role": "assistant", "content": str(row[output_col])})

        records.append({"messages": messages})

    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def prepare_training_data(data_dir: Path, cache_dir: Path, valid_split: float) -> Path:
    """Load parquet data and convert to JSONL format for training."""
    df = load_parquet_dataset(data_dir)
    print(f"Loaded {len(df)} rows")

    instruction_col, output_col, system_col = detect_columns(df)
    records = convert_to_chat_format(df, instruction_col, output_col, system_col)

    random.shuffle(records)

    if valid_split > 0 and len(records) > 1:
        split_idx = max(1, int(len(records) * (1 - valid_split)))
        train_records = records[:split_idx]
        valid_records = records[split_idx:]
    else:
        train_records = records
        valid_records = records  # Use train data for validation when no split

    cache_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_records, cache_dir / "train.jsonl")
    print(
        f"Wrote {len(train_records)} training examples to {cache_dir / 'train.jsonl'}"
    )

    write_jsonl(valid_records, cache_dir / "valid.jsonl")
    if valid_split > 0:
        print(
            f"Wrote {len(valid_records)} validation examples to {cache_dir / 'valid.jsonl'}"
        )
    else:
        print("Using training data for validation (no split specified)")

    return cache_dir, len(train_records)


def _default_adapter_path(experiment_name: str | None, model: str) -> Path:
    """Generate a unique adapter path from the experiment name (or model slug) and timestamp."""
    slug = experiment_name or model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("adapters") / f"{slug}_{timestamp}"


def train(
    model: str,
    data_dir: Path,
    adapter_path: Path | None,
    experiment_name: str | None,
    iters: int,
    batch_size: int,
    num_layers: int,
    learning_rate: float,
    valid_split: float,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str,
    grad_checkpoint: bool = False,
    max_seq_length: int = 2048,
    steps_per_eval: int = 50,
) -> None:
    """Run LoRA fine-tuning on the specified model."""
    if adapter_path is None:
        adapter_path = _default_adapter_path(experiment_name, model)
        print(f"Auto-generated adapter path: {adapter_path}")

    print(f"Fine-tuning model: {model}")

    cache_dir, num_train = prepare_training_data(data_dir, CACHE_DIR, valid_split)

    # Auto-adjust batch size for small datasets
    effective_batch_size = min(batch_size, num_train)
    if effective_batch_size < batch_size:
        print(
            f"Auto-adjusting batch size from {batch_size} to {effective_batch_size} (dataset has {num_train} examples)"
        )

    lora_parser = build_lora_parser()
    lora_args = lora_parser.parse_args(
        [
            "--model",
            model,
            "--data",
            str(cache_dir),
            "--train",
            "--iters",
            str(iters),
            "--batch-size",
            str(effective_batch_size),
            "--num-layers",
            str(num_layers),
            "--learning-rate",
            str(learning_rate),
            "--adapter-path",
            str(adapter_path),
            "--seed",
            "42",
            "--fine-tune-type",
            "lora",
            "--val-batches",
            "-1",
            "--max-seq-length",
            str(max_seq_length),
            "--steps-per-eval",
            str(steps_per_eval),
        ]
        + (["--grad-checkpoint"] if grad_checkpoint else [])
    )

    # Apply defaults from mlx_lm for any unset parameters
    for key, value in LORA_DEFAULTS.items():
        if not hasattr(lora_args, key) or getattr(lora_args, key) is None:
            setattr(lora_args, key, value)

    print(f"Adapter output: {adapter_path}")

    callback = None
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        print(f"MLflow tracking: {mlflow_tracking_uri}")
        print(f"MLflow experiment: {mlflow_experiment}")

        mlflow.start_run()
        mlflow.log_params(
            {
                "model": model,
                "adapter_path": str(adapter_path),
                "iters": iters,
                "batch_size": batch_size,
                "num_layers": num_layers,
                "learning_rate": learning_rate,
                "valid_split": valid_split,
            }
        )
        callback = MLflowCallback()

    print()

    # lora_run ignores its training_callback parameter — it overwrites it with
    # get_reporting_callbacks(args.report_to). Patch that function so our
    # MLflowCallback is used instead.
    _orig_get_reporting_callbacks = _mlx_lora.get_reporting_callbacks
    if callback is not None:
        _mlx_lora.get_reporting_callbacks = lambda *a, **kw: callback

    try:
        lora_run(lora_args)
        print(f"\nTraining complete! Adapter saved to {adapter_path}")
    finally:
        _mlx_lora.get_reporting_callbacks = _orig_get_reporting_callbacks
        if mlflow_tracking_uri:
            mlflow.end_run()


def test_model(model: str, adapter_path: Path, prompt: str, max_tokens: int) -> None:
    """Test the fine-tuned model with a prompt."""
    print(f"Loading model with adapter from {adapter_path}...")

    model_obj, tokenizer = load(model, adapter_path=str(adapter_path))

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(f"\nPrompt: {prompt}")
    print("\nResponse:", end=" ")

    response = generate(
        model_obj,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    print(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune instruct models using MLX with LoRA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Fine-tune a model")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="Experiment config file (e.g. configs/mentor_style_local.yaml); "
        "CLI flags override config values",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID or local path",
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing parquet files",
    )
    train_parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Output path for LoRA adapter "
        "(default: adapters/<experiment_name>_<timestamp>)",
    )
    train_parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of training iterations",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    train_parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers to apply LoRA to (-1 for all)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--valid-split",
        type=float,
        default=None,
        help="Fraction of data to use for validation (0 to disable)",
    )
    train_parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help=f"MLflow tracking server URI (default: {DEFAULT_MLFLOW_URI}, use 'none' to disable)",
    )
    train_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name",
    )

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Test a fine-tuned model")
    test_parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
        help="HuggingFace model ID or local path",
    )
    test_parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("adapters"),
        help="Path to LoRA adapter",
    )
    test_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to test the model with",
    )
    test_parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()

    if args.command == "train":
        if not args.data_dir.exists():
            print(f"Error: Data directory not found at {args.data_dir}")
            return

        cfg = load_config(args.config) if args.config else None
        student_cfg = cfg.student if cfg else None
        mlflow_cfg = cfg.mlflow if cfg else None

        # CLI flag > config value > hardcoded default
        model = args.model or (
            student_cfg.model if student_cfg else "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
        )
        iters = args.iters if args.iters is not None else (student_cfg.iters if student_cfg else 100)
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else (student_cfg.batch_size if student_cfg else 4)
        )
        num_layers = (
            args.num_layers
            if args.num_layers is not None
            else (student_cfg.num_layers if student_cfg else 16)
        )
        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else (student_cfg.learning_rate if student_cfg else 1e-5)
        )
        valid_split = (
            args.valid_split
            if args.valid_split is not None
            else (student_cfg.valid_split if student_cfg else 0.1)
        )
        mlflow_uri_raw = args.mlflow_tracking_uri or (
            mlflow_cfg.tracking_uri if mlflow_cfg else DEFAULT_MLFLOW_URI
        )
        mlflow_uri = None if mlflow_uri_raw.lower() == "none" else mlflow_uri_raw
        mlflow_exp = args.mlflow_experiment or (mlflow_cfg.experiment if mlflow_cfg else "finetune")
        experiment_name = cfg.experiment_name if cfg else None

        grad_checkpoint = student_cfg.grad_checkpoint if student_cfg else False
        max_seq_length = student_cfg.max_seq_length if student_cfg else 2048
        steps_per_eval = student_cfg.steps_per_eval if student_cfg else 50

        train(
            model=model,
            data_dir=args.data_dir,
            adapter_path=args.adapter_path,
            experiment_name=experiment_name,
            iters=iters,
            batch_size=batch_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            valid_split=valid_split,
            mlflow_tracking_uri=mlflow_uri,
            mlflow_experiment=mlflow_exp,
            grad_checkpoint=grad_checkpoint,
            max_seq_length=max_seq_length,
            steps_per_eval=steps_per_eval,
        )
    elif args.command == "test":
        if not args.adapter_path.exists():
            print(f"Error: Adapter not found at {args.adapter_path}")
            return
        test_model(args.model, args.adapter_path, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
