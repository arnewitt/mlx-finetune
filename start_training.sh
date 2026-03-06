#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/customer_support_style_nim.yaml"

echo "Generating training data..."
uv run python generate_data.py --config "$CONFIG"

echo "Starting training..."
uv run python main.py train --config "$CONFIG" --iters 200

echo "Done."
