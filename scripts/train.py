"""Train the FP32 or quantized CNN on Fashion-MNIST.

Examples (run inside the FINN Docker container, from the repo root):
    python scripts/train.py --model fp32
    python scripts/train.py --model w4a4
    python scripts/train.py --model w2a2

Writes runs/<model>/model.pt and runs/<model>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmnist_finn.data import load_split, to_float  # noqa: E402
from fmnist_finn.models import build_model, count_parameters  # noqa: E402


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1000) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            pred = model(x[i : i + batch_size]).argmax(dim=1)
            correct += (pred == y[i : i + batch_size]).sum().item()
    return correct / len(x)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="w4a4", help="fp32, w4a4, w2a2, ...")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 (fp32), 5e-3 (quantized)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=None, help="use only N training images (smoke tests)")
    p.add_argument("--out", default=None, help="output dir (default runs/<model>)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out) if args.out else REPO / "runs" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    xtr, ytr = load_split("train")
    xte, yte = load_split("test")
    if args.limit_train:
        xtr, ytr = xtr[: args.limit_train], ytr[: args.limit_train]
    xtr_t, ytr_t = torch.from_numpy(to_float(xtr)), torch.from_numpy(ytr)
    xte_t, yte_t = torch.from_numpy(to_float(xte)), torch.from_numpy(yte)
    loader = DataLoader(
        TensorDataset(xtr_t, ytr_t),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = build_model(args.model)
    quantized = args.model != "fp32"
    lr = args.lr if args.lr is not None else (5e-3 if quantized else 1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training {args.model}: {count_parameters(model)} params, {len(xtr)} train images, lr={lr}")
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            if quantized:
                model.clip_weights(-1.0, 1.0)
            running += loss.item() * len(xb)
            n += len(xb)
        acc = evaluate(model, xte_t, yte_t)
        history.append({"epoch": epoch + 1, "train_loss": running / n, "test_acc": acc})
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {running / n:.4f}  test acc {acc * 100:.2f}%  ({time.time() - t0:.0f}s)")

    torch.save(model.state_dict(), out_dir / "model.pt")
    metrics = {
        "model": args.model,
        "test_accuracy": history[-1]["test_acc"],
        "test_images": len(xte),
        "train_images": len(xtr),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "seed": args.seed,
        "parameters": count_parameters(model),
        "train_time_s": round(time.time() - t0, 1),
        "history": history,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "result_type": "measured (software)",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {out_dir / 'model.pt'}  test accuracy {metrics['test_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
