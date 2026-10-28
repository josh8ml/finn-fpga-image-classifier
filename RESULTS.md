# Results: Fashion-MNIST quantized CNN, CPU vs. FINN

> **Status: pending.** The pipeline has been tested end to end on a short smoke run. The numbers below get filled in from `results/summary.md` after the full runs on my machine. Nothing in this file is estimated or typed in by hand.

## Setup

| Item | Value |
|---|---|
| Dataset | Fashion-MNIST, 60k train / 10k test, 28×28 grayscale, 10 classes |
| Model | 4×Conv3×3 (16-16-32-32) + 2×MaxPool + FC 512→64→10, about 50k parameters, 2.51M MACs/image |
| Variants | FP32 (PyTorch), W4A4 and W2A2 (Brevitas quantization-aware training, 8-bit input) |
| Training | 10 epochs, Adam, cosine LR, batch 128, seed 0 |
| FPGA target | AUP-ZU3 (`xczu3eg-sfvc784-2-e`), 100 MHz |
| Tools | FINN `v1.0.0-alpha`, Vivado / Vitis HLS 2024.2 |

## 1. Accuracy (measured, software)

| Model | Test accuracy | Export matches Brevitas | Accuracy after FINN transformation |
|---|---:|---|---|
| FP32 | pending | n/a | n/a |
| W4A4 | pending | pending | pending |
| W2A2 | pending | pending | pending |

## 2. Baseline vs. FINN

| | CPU (FP32 PyTorch) | FINN W4A4 on AUP-ZU3 | FINN W2A2 on AUP-ZU3 |
|---|---|---|---|
| Accuracy | pending (measured) | pending (measured, software) | pending (measured, software) |
| Throughput | pending (measured) | pending (estimate) / pending (RTL simulation) | pending (estimate) |
| Latency | pending (measured, batch 1) | pending (estimate) / pending (RTL simulation) | pending (estimate) |
| LUT / BRAM / DSP | n/a | pending (estimate) / pending (post-route) | pending (estimate) |
| Timing at 100 MHz | n/a | pending (post-route) | not built |
| Power | not measured | pending (Vivado estimate) | not built |

The CPU numbers are measured on a desktop CPU inside Docker/WSL2. The FPGA throughput numbers are estimates or simulations of the accelerator pipeline and exclude host and DMA overheads. They are not on-board measurements, so this table isn't a like-for-like benchmark.

## 3. Throughput vs. area (estimate)

pending: from `finn_build/sweep.py`.

## Limitations

- No FPGA board yet, so there are no on-board throughput, latency, accuracy or power measurements.
- Power figures are Vivado's vectorless estimates.
- CPU timings depend on the machine, its load and the thread count, and run inside Docker on WSL2.
