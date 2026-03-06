#!/usr/bin/env python3
"""Download MLX models from Hugging Face to a local folder."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path("models")

# Popular MLX instruct models for quick reference
POPULAR_MODELS = [
    "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.3-70B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/gemma-2-9b-it-4bit",
    "mlx-community/phi-4-4bit",
]


def download_model(model_id: str, output_dir: Path) -> Path:
    """Download a model snapshot from Hugging Face to a local directory.

    The model files are placed under ``output_dir/<org>--<model-name>/``,
    mirroring the Hugging Face ``org/model`` path with ``/`` replaced by ``--``.

    Args:
        model_id: Hugging Face model repository ID (e.g. ``mlx-community/Llama-3.2-1B-Instruct-4bit``).
        output_dir: Parent directory under which the model folder is created.

    Returns:
        Path to the downloaded model directory.

    Raises:
        SystemExit: On network or disk errors, with a descriptive message.
    """
    model_name = model_id.replace("/", "--")
    local_path = output_dir / model_name

    print(f"Downloading {model_id}...")
    print(f"Destination: {local_path}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"Error: failed to download {model_id!r} — {e}")
        raise SystemExit(1) from e

    print(f"\nDownload complete: {local_path}")
    return local_path


def list_models(models_dir: Path) -> None:
    """Print the names of all downloaded model directories.

    Args:
        models_dir: Directory to scan for downloaded model folders.
    """
    if not models_dir.exists():
        print("No models downloaded yet.")
        return

    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        print("No models downloaded yet.")
        return

    print("Downloaded models:")
    for model_dir in sorted(model_dirs):
        print(f"  {model_dir.name}")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate sub-command."""
    parser = argparse.ArgumentParser(
        description="Download MLX models from Hugging Face"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download subcommand
    dl_parser = subparsers.add_parser("download", help="Download a model")
    dl_parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID (e.g., mlx-community/Llama-3.2-1B-Instruct-4bit)",
    )
    dl_parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory to save models",
    )

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List downloaded models")
    list_parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing models",
    )

    # Popular subcommand
    subparsers.add_parser("popular", help="Show popular MLX instruct models")

    args = parser.parse_args()

    if args.command == "download":
        args.output_dir.mkdir(exist_ok=True)
        download_model(args.model_id, args.output_dir)
    elif args.command == "list":
        list_models(args.output_dir)
    elif args.command == "popular":
        print("Popular MLX instruct models:")
        for model in POPULAR_MODELS:
            print(f"  {model}")


if __name__ == "__main__":
    main()
