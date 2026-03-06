# mlx-finetune

Fine-tune small instruct models on Apple Silicon using MLX and LoRA. Includes a teacher-student pipeline: a large teacher model generates synthetic training data, then a small student model is fine-tuned on it.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4/M5)
- Python 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone <repo>
cd mlx-finetune
uv sync
```

## Workflow

```
seeds/           →   generate_data.py   →   data/*.parquet   →   main.py train
(your prompts)       (teacher model)        (training data)       (student model)
```

### 1. Add seed prompts

Edit the text files in `seeds/` — one prompt per line. These are the topics the teacher will generate responses for.

```
# seeds/my_topic.txt
What is machine learning?
How does recursion work?
Explain gradient descent.
```

### 2. Add few-shot examples (optional but recommended)

Create a JSONL file in `examples/` to show the teacher the exact style, tone, and length you want:

```jsonl
{"instruction": "What is an API?", "output": "An API is a menu for software..."}
{"instruction": "Explain recursion.", "output": "Imagine two mirrors facing each other..."}
```

A ready-to-use set of 32 warm-mentor style examples ships in `examples/warm_mentor.jsonl`.

### 3. Configure an experiment

Create a YAML in `configs/` (or edit an existing one). It only needs to override keys that differ from `configs/base.yaml`:

```yaml
# configs/my_experiment.yaml
experiment_name: "my-experiment"

teacher:
  backend: "openai_api"         # local_mlx | openai_api
  model: "gpt-4o"
  examples_file: "examples/warm_mentor.jsonl"
  max_examples: 5

seeds:
  files:
    - "seeds/my_topic.txt"
  samples_per_seed: 3

generation:
  output_file: "my_data.parquet"

student:
  model: "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
  iters: 200
  batch_size: 1
  num_layers: 4
```

The `experiment_name` is used to name the saved adapter directory.

### 4. Generate training data

```bash
# Preview what will be generated (no API calls)
uv run python generate_data.py --config configs/my_experiment.yaml --dry-run

# Run generation
uv run python generate_data.py --config configs/my_experiment.yaml
```

Generation is **resumable** — if interrupted, re-running skips seeds already in the parquet file.

### 5. Fine-tune the student model

```bash
uv run python main.py train --config configs/my_experiment.yaml
```

The adapter is saved to `adapters/<experiment_name>_<timestamp>/` automatically. Each run gets its own directory so nothing is ever overwritten.

CLI flags override config values for quick tweaks:

```bash
uv run python main.py train --config configs/my_experiment.yaml --iters 50
```

### 6. Test the result

```bash
uv run python main.py test \
  --adapter-path adapters/my-experiment_20260302_143022 \
  --prompt "What is machine learning?"
```

## Teacher Backends

| Backend | Config value | Requirement |
|---|---|---|
| Local MLX model | `local_mlx` | Downloaded model in `models/` |
| OpenAI API | `openai_api` | `OPENAI_API_KEY` env var |
| NVIDIA NIM (cloud) | `openai_api` + `base_url` | `NVIDIA_API_KEY` env var |
| Ollama (local) | `openai_api` + `base_url` | `ollama serve` running |
| LM Studio | `openai_api` + `base_url` | LM Studio server running |
| vLLM / Groq / Together | `openai_api` + `base_url` | Endpoint URL |

### Local MLX teacher

```bash
uv run python download_model.py download mlx-community/Qwen2.5-7B-Instruct-4bit
```

```yaml
teacher:
  backend: "local_mlx"
  model: "mlx-community/Qwen2.5-7B-Instruct-4bit"
```

### OpenAI-compatible teacher

```bash
export OPENAI_API_KEY=sk-...
```

```yaml
teacher:
  backend: "openai_api"
  model: "gpt-4o"
```

For a local server (Ollama, LM Studio, vLLM):

```yaml
teacher:
  backend: "openai_api"
  model: "qwen3:8b"
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
```

### NVIDIA NIM (cloud)

```bash
export NVIDIA_API_KEY=nvapi-...
```

```yaml
teacher:
  backend: "openai_api"
  model: "meta/llama-3.3-70b-instruct"
  base_url: "https://integrate.api.nvidia.com/v1"
```

New accounts get 1,000 free credits at [build.nvidia.com](https://build.nvidia.com).

## Model Management

```bash
uv run python download_model.py popular     # list recommended MLX models
uv run python download_model.py download mlx-community/Qwen3.5-0.8B-OptiQ-4bit
uv run python download_model.py list        # list downloaded models
```

## Training Options

The `--config` flag is the primary way to configure training. All values in the `student:` and `mlflow:` config sections become defaults; any CLI flag overrides its config counterpart.

```bash
# Config-driven (recommended)
uv run python main.py train --config configs/my_experiment.yaml

# CLI-only (no config file needed)
uv run python main.py train \
  --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
  --data-dir data/ \
  --iters 200 \
  --batch-size 1 \
  --num-layers 4 \
  --learning-rate 1e-5 \
  --valid-split 0.1

# Override a single config value
uv run python main.py train --config configs/my_experiment.yaml --iters 50
```

| Flag | Default | Description |
|---|---|---|
| `--config` | — | Experiment YAML; sets defaults for all flags below |
| `--model` | `Qwen3.5-0.8B-OptiQ-4bit` | HuggingFace ID or local path |
| `--data-dir` | `data/` | Directory with parquet files |
| `--adapter-path` | `adapters/<name>_<timestamp>` | Where to save the LoRA adapter |
| `--iters` | `100` | Training iterations |
| `--batch-size` | `4` | Auto-reduced for small datasets |
| `--num-layers` | `16` | Layers to apply LoRA to (-1 = all) |
| `--learning-rate` | `1e-5` | AdamW learning rate |
| `--valid-split` | `0.1` | Validation fraction (0 = skip) |
| `--mlflow-tracking-uri` | `http://localhost:5002` | MLflow server (`none` to disable) |
| `--mlflow-experiment` | `finetune` | MLflow experiment name |

Defaults in the table apply when no `--config` is provided. When a config is given, the `student:` and `mlflow:` sections take precedence.

### Memory-constrained Macs

If you hit an out-of-memory error during training, reduce batch size and LoRA layer count in your config's `student:` section:

```yaml
student:
  batch_size: 1
  num_layers: 4
```

You can also tune how often validation runs (default: every 50 steps):

```yaml
student:
  steps_per_eval: 25
```

## Adapter Management

Each training run saves its adapter to a unique timestamped directory:

```
adapters/
  my-experiment_20260302_143022/              # from --config (uses experiment_name)
  my-experiment_20260302_160511/              # second run of the same config
  Qwen3.5-0.8B-OptiQ-4bit_20260302_091234/   # no config, named by model
```

To pin an adapter path (skips auto-generation):

```bash
uv run python main.py train --config configs/my_experiment.yaml --adapter-path adapters/fixed-name
```

## MLflow

Training metrics are logged to MLflow automatically.

```bash
# Start a local MLflow server
uv run mlflow ui --port 5002

# Use a different server or experiment
uv run python main.py train --config configs/my_experiment.yaml \
  --mlflow-tracking-uri http://localhost:8080 \
  --mlflow-experiment my-experiment

# Disable MLflow
uv run python main.py train --mlflow-tracking-uri none
```

**Logged metrics:** `train_loss`, `val_loss`, `learning_rate`, `tokens_per_second`, `iterations_per_second`, `trained_tokens`, `peak_memory_gb`

**Logged parameters:** `model`, `adapter_path`, `iters`, `batch_size`, `num_layers`, `learning_rate`, `valid_split`

## Data Quality

Generated data is automatically filtered before saving:

- **Length bounds** — drops outputs that are too short or too long
- **Refusal detection** — removes safety-filtered outputs
- **Exact deduplication** — removes identical outputs
- **Semantic deduplication** (optional) — removes near-duplicates by cosine similarity

```bash
# Enable semantic dedup (requires extra dependency)
uv add sentence-transformers
```

```yaml
quality:
  dedup_strategy: "semantic"
  semantic_threshold: 0.92
```

## Project Structure

```
mlx-finetune/
├── generate_data.py         # Generate synthetic data with a teacher model
├── main.py                  # Fine-tune and test the student model
├── download_model.py        # Download MLX models from Hugging Face
│
├── configs/                 # Experiment configs (YAML)
│   ├── base.yaml                      # Shared defaults
│   ├── mentor_style_local.yaml        # Local Qwen2.5-7B teacher
│   ├── mentor_style_openai.yaml       # OpenAI API teacher
│   ├── mentor_style_ollama.yaml       # Ollama (local) teacher
│   ├── mentor_style_nim.yaml          # NVIDIA NIM cloud teacher
│   └── style_transfer_small.yaml      # Small style-transfer experiment
│
├── seeds/                   # Seed prompts (one per line)
│   ├── general_knowledge.txt
│   └── style_transfer.txt
│
├── examples/                # Few-shot example pairs (JSONL)
│   └── warm_mentor.jsonl
│
├── finetune/                # Internal library
│   ├── config.py
│   ├── teacher/             # local_mlx and openai_api backends
│   ├── generation/          # Seed loading, examples, quality filtering
│   └── data/                # Parquet storage
│
├── data/                    # Generated parquet datasets
├── models/                  # Downloaded MLX models
├── adapters/                # LoRA adapters (one subdirectory per run)
└── .cache/                  # Auto-generated JSONL (train.jsonl, valid.jsonl)
```
