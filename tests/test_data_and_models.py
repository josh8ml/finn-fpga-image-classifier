"""Fast tests: dataset integrity, model shapes, quantization and operation counts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmnist_finn.data import CLASS_NAMES, FILES, load_split, md5sum  # noqa: E402
from fmnist_finn.models import build_model, count_macs  # noqa: E402


def test_dataset_checksums_and_shapes():
    x, y = load_split("test")  # downloads + verifies MD5 if needed
    assert x.shape == (10000, 1, 28, 28) and x.dtype == np.uint8
    assert y.shape == (10000,) and set(np.unique(y)) == set(range(10))
    assert len(CLASS_NAMES) == 10
    data_dir = Path(__file__).resolve().parents[1] / "data" / "fashion"
    for name, expected in FILES.items():
        assert md5sum(data_dir / name) == expected


@pytest.mark.parametrize("kind", ["fp32", "w4a4", "w2a2"])
def test_forward_shape(kind):
    model = build_model(kind).eval()
    with torch.no_grad():
        out = model(torch.rand(3, 1, 28, 28))
    assert out.shape == (3, 10)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("bits", [2, 4])
def test_weights_are_quantized(bits):
    """Each quantized layer's effective weights use at most 2**bits - 1 levels (narrow range)."""
    from brevitas.nn import QuantConv2d, QuantLinear

    model = build_model(f"w{bits}a{bits}").eval()
    layers = [m for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear))]
    assert len(layers) == 6
    for layer in layers:
        levels = torch.unique(layer.quant_weight().value.detach())
        assert len(levels) <= 2**bits - 1


def test_mac_count():
    macs = count_macs()
    # Cross-checked against FINN's op_and_param_counts.json report
    assert macs["conv0"] == 97_344
    assert macs["conv1"] == 1_327_104
    assert macs["total"] == 2_508_480
