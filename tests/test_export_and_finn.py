"""Integration tests: QONNX export equivalence and FINN conversion to hardware layers.

These run the real scripts on an untrained (randomly initialized) W2A2 model in a
temporary folder. They need the FINN environment (Brevitas, QONNX, FINN) but not
Vivado. Takes about a minute.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmnist_finn.models import build_model  # noqa: E402

HW_OP_PREFIXES = (
    "Thresholding",
    "ConvolutionInputGenerator",
    "MVAU",
    "Pool",
    "LabelSelect",
    "StreamingDataWidthConverter",
    "StreamingFIFO",
)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("w2a2_run")
    torch.manual_seed(0)
    torch.save(build_model("w2a2").state_dict(), d / "model.pt")
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_qonnx.py"), "--model", "w2a2", "--run-dir", str(d),
         "--check-images", "20"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return d


def test_export_matches_brevitas(run_dir):
    check = json.loads((run_dir / "export_check.json").read_text())
    assert check["allclose_atol_1e-4"]
    assert check["top1_agreement"] == 1.0


def test_finn_estimate_build_maps_everything_to_hardware(run_dir):
    r = subprocess.run(
        [sys.executable, str(REPO / "finn_build" / "build.py"), "--model", "w2a2", "--mode", "estimate",
         "--run-dir", str(run_dir)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = run_dir / "finn_estimate"

    # FINN's own verification steps compare each transformed graph against the golden output
    verif = sorted(p.name for p in (out / "verification_output").glob("*.npy"))
    assert verif and all("SUCCESS" in v for v in verif), verif

    # every node of the final graph is a hardware layer (nothing left to run in software)
    from qonnx.core.modelwrapper import ModelWrapper

    model = ModelWrapper(str(out / "intermediate_models" / "step_generate_estimate_reports.onnx"))
    ops = [n.op_type for n in model.graph.node]
    assert ops and all(op.startswith(HW_OP_PREFIXES) for op in ops), ops

    perf = json.loads((out / "report" / "estimate_network_performance.json").read_text())
    assert perf["estimated_throughput_fps"] > 0
