# FPGA-Accelerated Quantized Image Classifier Using FINN

> **Work in progress.** This repository documents my work with [FINN](https://github.com/Xilinx/finn), AMD Research's open-source compiler that turns quantized neural networks into FPGA dataflow accelerators. The goal: train a small quantized image classifier, compare it with a CPU baseline, and generate, simulate and synthesize a FINN hardware implementation.

## Status

| Stage | Status |
|---|---|
| FINN environment on Windows (WSL2 + Docker) | Done: FINN's software test suite passes |
| FINN tutorials (ONNX basics, Brevitas → QONNX → FINN import) | Done |
| Vivado/Vitis 2024.2 connected to FINN (Vitis HLS + RTL simulation verified) | Done |
| FINN end-to-end tutorial built to a bitstream for the AUP-ZU3 (no board) | Done: timing met at 100 MHz |
| Own project: quantized image classifier + CPU baseline | Planned |
| FINN estimate / RTL simulation / synthesis results for own project | Planned |
| On-board measurements | Needs an FPGA board (AUP-ZU3 or Kria KV260) |
| Upstream contribution to FINN | In progress |

## Verified results so far

All results below were actually run. None were measured on an FPGA board yet.

| What | Result | Type |
|---|---|---|
| FINN `v1.0.0-alpha` (`38c42bf`) non-Vivado test suite (`./run-docker.sh quicktest`) | 2736 passed, 16 skipped, 5 xfailed, 1 xpassed | Software tests |
| `basics/0_how_to_work_with_onnx` | Graph transformation output matches NumPy reference | Software |
| `basics/1_brevitas_network_import_via_QONNX` | Quantized MNIST MLP gives identical outputs in Brevitas (PyTorch), QONNX and FINN-ONNX | Software |
| FINN `test_fpgadataflow_fmpadding` (rtlsim, AUP-ZU3 part `xczu3eg`) | Passed; simulated cycle count within 10 cycles of FINN's analytical estimate | RTL simulation |
| FINN `test_fpgadataflow_checksum` (AUP-ZU3 part `xczu3eg`) | Passed: HLS layers synthesized with Vitis HLS 2024.2, verified in C++ and RTL simulation | HLS + RTL simulation |
| FINN `tfc_end2end_example`: binarized MNIST MLP → bitstream for AUP-ZU3 (`xczu3eg`) | Timing met at 100 MHz (WNS +3.81 ns). Accelerator: 8,246 LUTs (11.7%), 10,853 FFs (7.7%), 11 BRAM36 (5.1%), 0 DSPs. Total on-chip power 2.87 W (Vivado estimate, mostly the ARM PS) | Vivado post-route reports ([details](docs/results-tfc-end2end.md)) |

**Results policy:** every number in this repo is labelled as *measured (software)*, *estimate* (FINN analytical model), *simulation* (RTL sim), *synthesis report* (Vivado), or *measured (hardware)*. No FPGA resource, timing, throughput or power figures are claimed without the corresponding run.

## Documentation

- [`docs/PROJECT_NOTES.md`](docs/PROJECT_NOTES.md): how FINN works (Brevitas → QONNX → FINN-ONNX → HLS/RTL → Vivado), the dataflow architecture, PE/SIMD folding, and how throughput, latency, LUTs, BRAM, DSPs, clock and power relate.
- [`docs/results-tfc-end2end.md`](docs/results-tfc-end2end.md): post-route timing, utilization (whole design vs. accelerator, per layer) and power for FINN's end-to-end tutorial on the AUP-ZU3.
- [`docs/setup-log.md`](docs/setup-log.md): every command, version, error and fix.
- [`docs/windows-wsl-setup.md`](docs/windows-wsl-setup.md): reproducible setup guide for running FINN on Windows via WSL2.

## Tools

FINN `v1.0.0-alpha` · Brevitas · QONNX/ONNX · PyTorch · Docker · WSL2 (Ubuntu) · Vivado / Vitis HLS 2024.2

## Acknowledgements

Built on [FINN](https://github.com/Xilinx/finn) (AMD Research & Advanced Development, BSD-3-Clause). This repository is my own learning project and is not affiliated with AMD.
