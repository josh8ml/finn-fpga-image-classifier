# FPGA-Accelerated Quantized Image Classifier Using FINN

> 🚧 **Work in progress.** This repository documents my work with [FINN](https://github.com/Xilinx/finn), AMD Research's open-source compiler that turns quantized neural networks into FPGA dataflow accelerators. The goal: train a small quantized image classifier, compare it with a CPU baseline, and generate, simulate and synthesize a FINN hardware implementation.

## Status

| Stage | Status |
|---|---|
| FINN environment on Windows (WSL2 + Docker) | ✅ Done: FINN's software test suite passes |
| FINN tutorials (ONNX basics, Brevitas → QONNX → FINN import) | ✅ Done |
| Vivado/Vitis 2024.2 + FINN's Vivado verification | ⏳ In progress |
| Own project: quantized image classifier + CPU baseline | ⬜ Planned |
| FINN estimate / RTL simulation / synthesis results | ⬜ Planned |
| On-board measurements | ⬜ Needs an FPGA board (AUP-ZU3 or Kria KV260) |
| Upstream contribution to FINN | ⬜ Investigating |

## Verified results so far

All results below were actually run. Nothing here comes from FPGA hardware yet.

| What | Result | Type |
|---|---|---|
| FINN `v1.0.0-alpha` (`38c42bf`) non-Vivado test suite (`./run-docker.sh quicktest`) | 2736 passed, 16 skipped, 5 xfailed, 1 xpassed | Software tests |
| `basics/0_how_to_work_with_onnx` | Graph transformation output matches NumPy reference | Software |
| `basics/1_brevitas_network_import_via_QONNX` | Quantized MNIST MLP gives identical outputs in Brevitas (PyTorch), QONNX and FINN-ONNX | Software |

**Results policy:** every number in this repo is labelled as *measured (software)*, *estimate* (FINN analytical model), *simulation* (RTL sim), *synthesis report* (Vivado), or *measured (hardware)*. No FPGA resource, timing, throughput or power figures are claimed without the corresponding run.

## Documentation

- [`docs/PROJECT_NOTES.md`](docs/PROJECT_NOTES.md): how FINN works (Brevitas → QONNX → FINN-ONNX → HLS/RTL → Vivado), the dataflow architecture, PE/SIMD folding, and how throughput, latency, LUTs, BRAM, DSPs, clock and power relate.
- [`docs/setup-log.md`](docs/setup-log.md): every command, version, error and fix.
- [`docs/windows-wsl-setup.md`](docs/windows-wsl-setup.md): reproducible setup guide for running FINN on Windows via WSL2.

## Tools

FINN `v1.0.0-alpha` · Brevitas · QONNX/ONNX · PyTorch · Docker · WSL2 (Ubuntu) · Vivado/Vitis 2024.2 (in progress)

## Acknowledgements

Built on [FINN](https://github.com/Xilinx/finn) (AMD Research & Advanced Development, BSD-3-Clause). This repository is my own learning project and is not affiliated with AMD.
