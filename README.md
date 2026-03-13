# gesring-benchmark

The dataset and benchmark of gesring.

## Data Download

Download the dataset from Zenodo:

https://zenodo.org/records/18933199?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3MzEyMjI4MSwiZXhwIjoxNzc4MzcxMTk5fQ.eyJpZCI6ImU5NTlmMzU3LTA2NWYtNDI5ZC1hOWU4LTlhNDIxOWRmYTI2NyIsImRhdGEiOnt9LCJyYW5kb20iOiI2ZjM4NGJmMWM0MDdlNjA1MDAzZmE3ZThlODg5YzhhNyJ9.VqcnINJWgCUupibwIQnPB6X6938wIDG5w77hXcruK0msuVwlta8_3y1a2g74TkgZFwCSnEbHCuM3kq_yjEdahw

Recommended layout:

```text
gesring-benchmark/
├── README.md
├── LICENSE
├── benchmark/
│   └── eval_checkpoint.py
└── data/
    └── gesture_merged/
        ├── test_x.npy
        └── test_y.npy
```

## Data Format

Benchmarking uses two `.npy` files by default:

- `data/gesture_merged/test_x.npy`: IMU sequence, shape `N x 6 x 200`
- `data/gesture_merged/test_y.npy`: labels, shape `N`

Conventions:

- `N` is the number of samples
- `6` is the number of IMU channels
- `200` is the time window length
- recommended dtype: `x` as `float32`, `y` as `int64`

## Benchmark Evaluation Scheme (Checkpoint)

Evaluation uses a fixed test set (default `data/gesture_merged/test_x.npy` and `data/gesture_merged/test_y.npy`) plus your checkpoint.

### Checkpoint Specification

The evaluation script in this repo uses a **TorchScript checkpoint**:

- file format: `*.pt`
- model input: `float32` tensor with shape `[B, 6, 200]`
- model output supports two forms:
1. `[B, C]` logits/probabilities (the script applies `argmax`)
2. `[B]` predicted class ids

### Metrics

- `accuracy`
- `macro_f1`
- `confusion_matrix`

## Full Evaluation Workflow

### 1. Install dependencies with uv

This project is managed by `uv` and pinned by `uv.lock` (Python `3.12`). Install with lockfile:

```bash
uv sync --locked
```

Optional: activate the virtual environment for direct `python` usage:

```bash
source .venv/bin/activate
```

If you do not activate the environment, run commands with `uv run`.

### 2. Prepare data

Make sure the test files exist at the default paths:

```text
data/gesture_merged/test_x.npy
data/gesture_merged/test_y.npy
```

And they contain:

- `x` -> `N x 6 x 200`
- `y` -> `N`

### 3. Export a TorchScript checkpoint

If your current model is a standard PyTorch model, export it to TorchScript first:

```python
import torch

# model: your trained nn.Module
# model.load_state_dict(torch.load("your_state_dict.ckpt", map_location="cpu"))
model.eval()
example = torch.randn(1, 6, 200)
scripted = torch.jit.trace(model, example)
scripted.save("checkpoints/model.pt")
```

### 4. Run evaluation

```bash
uv run python benchmark/eval_checkpoint.py \
  --checkpoint checkpoints/model.pt \
  --batch-size 256 \
  --device cpu \
  --output results/test_metrics.json
```

If you have a GPU:

```bash
uv run python benchmark/eval_checkpoint.py \
  --checkpoint checkpoints/model.pt \
  --device cuda
```

### 5. View results

The script prints JSON in terminal and can also write to the file specified by `--output`. The output fields include:

- `num_samples`
- `num_classes`
- `accuracy`
- `macro_f1`
- `confusion_matrix`

## Evaluation Script

- `benchmark/eval_checkpoint.py`: unified evaluation entrypoint based on TorchScript checkpoints.
