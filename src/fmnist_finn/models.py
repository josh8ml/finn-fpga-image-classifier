"""Small CNN for Fashion-MNIST: a quantized (Brevitas) version for FINN and an FP32 twin.

The topology deliberately follows FINN's reference CNV network
(brevitas_examples.bnn_pynq.models.CNV), scaled down for 28x28 grayscale input:

    input 1x28x28
    conv3x3  1->16   26x26   BN  act
    conv3x3 16->16   24x24   BN  act   maxpool2 -> 12x12
    conv3x3 16->32   10x10   BN  act
    conv3x3 32->32    8x8    BN  act   maxpool2 ->  4x4
    flatten 512
    fc 512->64              BN  act
    fc  64->10              TensorNorm (scalar affine on the logits)

Same building blocks as CNV, so FINN's default build flow can compile it:
no-padding 3x3 convolutions without bias, BatchNorm, signed activations in
[-1, 1], 2x2 max-pooling only on even feature-map sizes, and the same
quantizer definitions (CommonWeightQuant / CommonActQuant).
"""

from __future__ import annotations

import torch
from torch import nn

CONV_CHANNELS_POOL = [(16, False), (16, True), (32, False), (32, True)]
FC_HIDDEN = 64
NUM_CLASSES = 10
IN_CHANNELS = 1
KERNEL_SIZE = 3
FLAT_FEATURES = 32 * 4 * 4  # after the second pool


def _preprocess(x: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1], as in CNV (the first quantizer then uses a Q1.7 format)
    return 2.0 * x - torch.tensor([1.0], device=x.device)


class QuantCNN(nn.Module):
    """Quantized CNN built from Brevitas layers (exportable to QONNX for FINN)."""

    def __init__(self, weight_bit_width: int = 4, act_bit_width: int = 4, in_bit_width: int = 8):
        super().__init__()
        # Imported here so the FP32 model and the data code work without Brevitas.
        from brevitas.core.restrict_val import RestrictValueType
        from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
        from brevitas_examples.bnn_pynq.models.common import CommonActQuant, CommonWeightQuant
        from brevitas_examples.bnn_pynq.models.tensor_norm import TensorNorm

        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.conv_features = nn.ModuleList()
        self.conv_features.append(
            QuantIdentity(  # 8-bit input quantizer, Q1.7 format (same as CNV)
                act_quant=CommonActQuant,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            )
        )
        in_ch = IN_CHANNELS
        for out_ch, pool in CONV_CHANNELS_POOL:
            self.conv_features.append(
                QuantConv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=KERNEL_SIZE,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            self.conv_features.append(nn.BatchNorm2d(out_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            if pool:
                self.conv_features.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch

        self.linear_features = nn.ModuleList(
            [
                QuantLinear(
                    FLAT_FEATURES,
                    FC_HIDDEN,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                ),
                nn.BatchNorm1d(FC_HIDDEN, eps=1e-4),
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width),
                QuantLinear(
                    FC_HIDDEN,
                    NUM_CLASSES,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                ),
                TensorNorm(),
            ]
        )

        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        """Keep float weights inside the quantizer's range (as in Brevitas' CNV training)."""
        from brevitas.nn import QuantConv2d, QuantLinear

        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                m.weight.data.clamp_(min_val, max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _preprocess(x)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


class FloatCNN(nn.Module):
    """FP32 twin of QuantCNN: same topology, ordinary PyTorch layers.

    Hardtanh stands in for the quantized activations (both are signed and
    limited to [-1, 1]). Used for the accuracy baseline and the CPU timing baseline.
    """

    def __init__(self):
        super().__init__()
        layers = []
        in_ch = IN_CHANNELS
        for out_ch, pool in CONV_CHANNELS_POOL:
            layers += [
                nn.Conv2d(in_ch, out_ch, KERNEL_SIZE, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-4),
                nn.Hardtanh(-1.0, 1.0),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.conv_features = nn.Sequential(*layers)
        self.linear_features = nn.Sequential(
            nn.Linear(FLAT_FEATURES, FC_HIDDEN, bias=False),
            nn.BatchNorm1d(FC_HIDDEN, eps=1e-4),
            nn.Hardtanh(-1.0, 1.0),
            nn.Linear(FC_HIDDEN, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _preprocess(x)
        x = self.conv_features(x)
        x = x.view(x.shape[0], -1)
        return self.linear_features(x)


def build_model(kind: str) -> nn.Module:
    """kind: 'fp32', or 'wXaY' (e.g. 'w4a4', 'w2a2')."""
    if kind == "fp32":
        return FloatCNN()
    if kind.startswith("w") and "a" in kind:
        w, a = kind[1:].split("a")
        return QuantCNN(weight_bit_width=int(w), act_bit_width=int(a))
    raise ValueError(f"Unknown model kind {kind!r} (use 'fp32', 'w4a4', 'w2a2', ...)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs() -> dict:
    """Multiply-accumulate operations per image for each compute layer."""
    macs = {}
    size, in_ch = 28, IN_CHANNELS
    for i, (out_ch, pool) in enumerate(CONV_CHANNELS_POOL):
        size -= KERNEL_SIZE - 1
        macs[f"conv{i}"] = size * size * out_ch * in_ch * KERNEL_SIZE * KERNEL_SIZE
        if pool:
            size //= 2
        in_ch = out_ch
    macs["fc0"] = FLAT_FEATURES * FC_HIDDEN
    macs["fc1"] = FC_HIDDEN * NUM_CLASSES
    macs["total"] = sum(macs.values())
    return macs
