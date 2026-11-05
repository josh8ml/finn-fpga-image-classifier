"""CPU baseline: latency and throughput of the FP32 model in PyTorch on this machine.

    python scripts/cpu_baseline.py

Reads runs/fp32/model.pt, writes runs/fp32/cpu_baseline.json.
Result type: measured (software), on the CPU this runs on. Note the CPU model and
thread count when comparing, and that it runs inside Docker/WSL2.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmnist_finn.data import load_split, to_float  # noqa: E402
from fmnist_finn.models import build_model  # noqa: E402


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=str(REPO / "runs" / "fp32"))
    p.add_argument("--latency-iters", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threads", type=int, default=None, help="torch threads (default: PyTorch's choice)")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    run_dir = Path(args.run_dir)
    model = build_model("fp32")
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()

    xte, yte = load_split("test")
    x = torch.from_numpy(to_float(xte))

    with torch.inference_mode():
        # batch-1 latency
        for i in range(50):
            model(x[i : i + 1])
        lat = []
        for i in range(args.latency_iters):
            xi = x[i % len(x) : i % len(x) + 1]
            t0 = time.perf_counter()
            model(xi)
            lat.append(time.perf_counter() - t0)
        # batched throughput over the full test set (also gives accuracy)
        model(x[: args.batch_size])
        t0 = time.perf_counter()
        preds = torch.cat([model(x[i : i + args.batch_size]).argmax(1) for i in range(0, len(x), args.batch_size)])
        elapsed = time.perf_counter() - t0

    lat_us = sorted(v * 1e6 for v in lat)
    result = {
        "result_type": "measured (software)",
        "model": "fp32",
        "cpu": cpu_name(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "batch1_latency_us_median": round(statistics.median(lat_us), 1),
        "batch1_latency_us_p99": round(lat_us[int(0.99 * len(lat_us)) - 1], 1),
        "batch1_throughput_img_per_s": round(1e6 / statistics.median(lat_us), 1),
        f"batch{args.batch_size}_throughput_img_per_s": round(len(x) / elapsed, 1),
        "test_accuracy": round((preds.numpy() == yte).mean(), 4),
        "test_images": len(x),
    }
    (run_dir / "cpu_baseline.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
