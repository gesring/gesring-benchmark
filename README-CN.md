# gesring-benchmark

The dataset and benchmark of gesring.

## 数据下载

从 Zenodo 下载数据集：

https://zenodo.org/records/18933199?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3MzEyMjI4MSwiZXhwIjoxNzc4MzcxMTk5fQ.eyJpZCI6ImU5NTlmMzU3LTA2NWYtNDI5ZC1hOWU4LTlhNDIxOWRmYTI2NyIsImRhdGEiOnt9LCJyYW5kb20iOiI2ZjM4NGJmMWM0MDdlNjA1MDAzZmE3ZThlODg5YzhhNyJ9.VqcnINJWgCUupibwIQnPB6X6938wIDG5w77hXcruK0msuVwlta8_3y1a2g74TkgZFwCSnEbHCuM3kq_yjEdahw

推荐放置方式：

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

## 数据格式

评测默认使用两个 `.npy` 文件：

- `data/gesture_merged/test_x.npy`: IMU 序列，形状为 `N x 6 x 200`
- `data/gesture_merged/test_y.npy`: 标签，形状为 `N`

约定：

- `N` 是样本数
- `6` 是 IMU 通道数
- `200` 是时间窗口长度
- `x` dtype 建议为 `float32`，`y` dtype 建议为 `int64`

## Benchmark 评测方案（Checkpoint）

评测使用固定测试集（默认 `data/gesture_merged/test_x.npy` 和 `data/gesture_merged/test_y.npy`）+ 你提供的 checkpoint。

### Checkpoint 规范

当前仓库提供的评测脚本使用 **TorchScript checkpoint**：

- 文件格式：`*.pt`
- 模型输入：`[B, 6, 200]` 的 `float32` tensor
- 模型输出支持两种形式：
1. `[B, C]` logits/probabilities（脚本内部取 `argmax`）
2. `[B]` 预测类别 id

### 评测指标

- `accuracy`
- `macro_f1`
- `confusion_matrix`

## 完整评测流程

### 1. 使用 uv 安装依赖

```bash
uv venv
source .venv/bin/activate
uv pip install numpy torch
```

### 2. 准备数据

确保测试集文件存在（默认路径）：

```text
data/gesture_merged/test_x.npy
data/gesture_merged/test_y.npy
```

并且包含：

- `x` -> `N x 6 x 200`
- `y` -> `N`

### 3. 导出 TorchScript checkpoint

如果你当前是普通 PyTorch 模型，可以先导出 TorchScript：

```python
import torch

# model: your trained nn.Module
# model.load_state_dict(torch.load("your_state_dict.ckpt", map_location="cpu"))
model.eval()
example = torch.randn(1, 6, 200)
scripted = torch.jit.trace(model, example)
scripted.save("checkpoints/model.pt")
```

### 4. 运行评测

```bash
uv run python benchmark/eval_checkpoint.py \
  --checkpoint checkpoints/model.pt \
  --batch-size 256 \
  --device cpu \
  --output results/test_metrics.json
```

如果有 GPU：

```bash
uv run python benchmark/eval_checkpoint.py \
  --checkpoint checkpoints/model.pt \
  --device cuda
```

### 5. 查看结果

脚本会在终端打印 JSON，并可写入 `--output` 指定文件。输出字段包括：

- `num_samples`
- `num_classes`
- `accuracy`
- `macro_f1`
- `confusion_matrix`

## 评测脚本

- `benchmark/eval_checkpoint.py`: 基于 TorchScript checkpoint 的统一评测入口。
