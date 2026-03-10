#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TorchScript checkpoint on gesture IMU data (x: N*6*200, y: N)."
    )
    parser.add_argument(
        "--x-path",
        type=Path,
        default=Path("data/gesture_merged/test_x.npy"),
        help="Path to test x .npy, expected shape [N, 6, 200]",
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=Path("data/gesture_merged/test_y.npy"),
        help="Path to test y .npy, expected shape [N]",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to TorchScript .pt checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_data(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path)
    y = np.load(y_path)

    if x.ndim != 3 or x.shape[1:] != (6, 200):
        raise ValueError(f"Expected x shape [N, 6, 200], got {x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"Expected y shape [N], got {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y sample count mismatch: {x.shape[0]} vs {y.shape[0]}.")

    return x.astype(np.float32), y.astype(np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1_scores = []
    for cls in range(cm.shape[0]):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp

        if tp == 0 and (fp > 0 or fn > 0):
            f1_scores.append(0.0)
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def infer_logits_or_labels(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 1:
        return output.to(torch.int64)
    if output.ndim == 2:
        return torch.argmax(output, dim=1)
    raise ValueError(f"Unsupported model output shape: {tuple(output.shape)}")


def run_inference(model: torch.jit.ScriptModule, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            out = model(xb)
            if not isinstance(out, torch.Tensor):
                raise ValueError("Model output must be a torch.Tensor.")
            pb = infer_logits_or_labels(out).detach().cpu().numpy()
            preds.append(pb)

    return np.concatenate(preds, axis=0)


def main() -> None:
    args = parse_args()
    x, y_true = load_data(args.x_path, args.y_path)

    device = torch.device(args.device)
    model = torch.jit.load(str(args.checkpoint), map_location=device)

    y_pred = run_inference(model, x, args.batch_size, device)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"Pred length mismatch: {y_pred.shape[0]} vs {y_true.shape[0]}.")

    n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)

    accuracy = float((y_pred == y_true).mean())
    macro_f1 = macro_f1_from_cm(cm)

    result = {
        "num_samples": int(y_true.shape[0]),
        "num_classes": n_classes,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
