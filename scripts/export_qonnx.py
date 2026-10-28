"""Export a trained quantized model to QONNX and check that the export is exact.

    python scripts/export_qonnx.py --model w4a4

Writes, into runs/<model>/:
  model_qonnx.onnx       QONNX graph (input of the FINN build)
  verify_input.npy       one test image, uint8 values in a float32 container, shape (1,1,28,28)
  verify_expected.npy    its top-1 class from Brevitas, shape (1,1)
  export_check.json      Brevitas-vs-QONNX comparison on N test images
"""

from __future__ import annotations

import argparse
import json
import sys
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
    p.add_argument("--run-dir", default=None, help="default runs/<model>")
    p.add_argument("--check-images", type=int, default=100, help="test images to compare after export")
    args = p.parse_args()
    if args.model == "fp32":
        sys.exit("Only quantized models (e.g. w4a4) can be exported for FINN.")

    from brevitas.export import export_qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "runs" / args.model
    model = build_model(args.model)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()

    onnx_path = run_dir / "model_qonnx.onnx"
    export_qonnx(model, torch.rand(1, 1, 28, 28), str(onnx_path))
    qonnx_cleanup(str(onnx_path), out_file=str(onnx_path))
    print(f"Exported {onnx_path}")

    # --- check: QONNX execution must reproduce Brevitas outputs ---
    xte, yte = load_split("test")
    x = to_float(xte[: args.check_images])
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    w = ModelWrapper(str(onnx_path))
    iname, oname = w.graph.input[0].name, w.graph.output[0].name
    got = np.concatenate([execute_onnx(w, {iname: x[i : i + 1]})[oname] for i in range(len(x))])
    max_abs_diff = float(np.abs(ref - got).max())
    same_top1 = float((ref.argmax(1) == got.argmax(1)).mean())
    check = {
        "images": len(x),
        "max_abs_logit_diff": max_abs_diff,
        "top1_agreement": same_top1,
        "allclose_atol_1e-4": bool(np.allclose(ref, got, atol=1e-4)),
        "result_type": "measured (software)",
    }
    (run_dir / "export_check.json").write_text(json.dumps(check, indent=2))
    print(f"Brevitas vs QONNX on {len(x)} images: max |diff| {max_abs_diff:.2e}, top-1 agreement {same_top1 * 100:.1f}%")
    if not check["allclose_atol_1e-4"]:
        sys.exit("ERROR: QONNX export does not match Brevitas.")

    # --- golden input/output for the FINN build's verification steps ---
    np.save(run_dir / "verify_input.npy", xte[:1].astype(np.float32))  # raw uint8 pixel values
    np.save(run_dir / "verify_expected.npy", ref[:1].argmax(1).reshape(1, 1).astype(np.float32))
    print(f"Saved verification I/O (test image 0, label {yte[0]}, predicted {ref[0].argmax()})")


if __name__ == "__main__":
    main()
