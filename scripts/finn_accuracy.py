"""Accuracy of the FINN-transformed model (integer-only, after streamlining).

    python scripts/finn_accuracy.py --model w4a4 --images 1000

Executes runs/<model>/finn_estimate/intermediate_models/step_streamline.onnx with
FINN's Python ONNX execution on raw uint8 test images, and compares its top-1
output with the true labels and with the Brevitas model's predictions.
Run finn_build/build.py --mode estimate first.

Result type: measured (software). This checks that FINN's graph transformations
(pre-processing, streamlining into integer thresholds, top-1) keep the accuracy.
It is not a hardware measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmnist_finn.data import load_split, to_float  # noqa: E402
from fmnist_finn.models import build_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="w4a4")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--images", type=int, default=1000, help="number of test images (10000 = all)")
    p.add_argument("--stage", default="step_streamline", help="intermediate model to execute")
    args = p.parse_args()

    from qonnx.core.modelwrapper import ModelWrapper

    import finn.core.onnx_exec as oxe

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "runs" / args.model
    onnx_path = run_dir / "finn_estimate" / "intermediate_models" / f"{args.stage}.onnx"
    if not onnx_path.exists():
        sys.exit(f"{onnx_path} not found: run finn_build/build.py --model {args.model} --mode estimate first")

    xte, yte = load_split("test")
    xte, yte = xte[: args.images], yte[: args.images]

    brevitas = build_model(args.model)
    brevitas.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    brevitas.eval()
    with torch.no_grad():
        ref = brevitas(torch.from_numpy(to_float(xte))).argmax(1).numpy()

    model = ModelWrapper(str(onnx_path))
    iname, oname = model.graph.input[0].name, model.graph.output[0].name
    t0 = time.time()
    finn_pred = np.empty(len(xte), dtype=np.int64)
    for i in range(len(xte)):
        out = oxe.execute_onnx(model, {iname: xte[i : i + 1].astype(np.float32)})[oname]
        finn_pred[i] = int(np.asarray(out).flatten()[0])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(xte)} images ({time.time() - t0:.0f}s)")

    result = {
        "result_type": "measured (software)",
        "stage": args.stage,
        "images": int(len(xte)),
        "finn_accuracy": round(float((finn_pred == yte).mean()), 4),
        "brevitas_accuracy_same_images": round(float((ref == yte).mean()), 4),
        "top1_agreement_finn_vs_brevitas": round(float((finn_pred == ref).mean()), 4),
    }
    (run_dir / "finn_accuracy.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
