# Results: FINN `tfc_end2end_example` on the AUP-ZU3 (Vivado 2024.2)

FINN's official end-to-end tutorial (`notebooks/end2end_example/bnn-pynq/tfc_end2end_example.ipynb`), run to completion on my machine: a binarized MNIST MLP compiled from Brevitas to a complete Zynq UltraScale+ bitstream.

**Result type: Vivado post-route implementation reports.** No board was used, so there are no on-board throughput, latency or power measurements. Power is Vivado's *estimate*.

| Item | Value |
|---|---|
| Network | TFC-w1a1: fully connected 784 → 64 → 64 → 64 → 10, 1-bit weights, 1-bit activations (pretrained, from the FINN tutorial) |
| Target | `AUP-ZU3_8GB` board, part **xczu3eg-sfvc784-2-e** |
| Tools | FINN `v1.0.0-alpha` (`38c42bf`), Vivado / Vitis HLS 2024.2, WSL2 + Docker |
| Flow | Brevitas → QONNX → FINN-ONNX → streamline → HLS layers → folding → `ZynqBuild` (HLS synthesis, IP stitching, Zynq shell with DMAs, synthesis, place & route, bitstream) |
| Clock target | 100 MHz (10 ns) |
| Notebook | All cells completed without errors |

## Timing (post-route)

| Metric | Value |
|---|---|
| Worst negative slack (setup, WNS) | **+3.812 ns** |
| Worst hold slack (WHS) | +0.010 ns |
| Failing endpoints | 0 of 80,139 |
| Status | **All user-specified timing constraints met** at 100 MHz |
| Routing | 40,259 / 40,259 nets fully routed, 0 routing errors |

The slack suggests the critical path is about 6.2 ns (≈160 MHz). That's only a rough indicator: the design wasn't re-implemented at a higher clock, so a higher Fmax is **not** a verified result.

## Resource utilization (post-route), whole design vs. FINN accelerator

The full design includes the Zynq shell (DMA engines, AXI interconnects, reset) around the FINN-generated accelerator (`StreamingDataflowPartition_1`).

| Resource | Available | Whole design | FINN accelerator only |
|---|---:|---:|---:|
| CLB LUTs | 70,560 | 17,681 (25.1%) | **8,246 (11.7%)** |
| CLB registers (FFs) | 141,120 | 28,314 (20.1%) | **10,853 (7.7%)** |
| Block RAM tiles (36 Kb) | 216 | 13.5 (6.3%) | **11 (5.1%)** |
| DSP slices | 360 | 0 | **0** |

### Per-layer breakdown inside the accelerator

| Layer | Folding (PE × SIMD) | LUTs | FFs | BRAM36 | Notes |
|---|---|---:|---:|---:|---|
| Thresholding (input quantizer) | PE = 49 | 361 | 532 | 0 | Binarizes the 784 input pixels |
| MVAU 0 (784 → 64) | 16 × 49 | 4,057 | 4,737 | 11 | Largest layer; 50,176 weight bits in block RAM (`ram_style="block"`) |
| MVAU 1 (64 → 64) | 8 × 8 | 871 | 1,384 | 0 | |
| MVAU 2 (64 → 64) | 8 × 8 | 869 | 1,376 | 0 | |
| MVAU 3 (64 → 10) | 10 × 8 | 1,153 | 1,777 | 0 | Weights in LUTRAM (`ram_style="distributed"`) |
| LabelSelect (top-1) | — | 125 | 134 | 0 | Outputs the predicted class |
| FIFOs + data-width converters | — | 815 | 913 | 0 | Inserted automatically by FINN |

Shell outside the accelerator: input DMA 3,014 LUTs, output DMA 1,004 LUTs, AXI SmartConnect 4,129 LUTs, AXI interconnect 1,276 LUTs.

## Power (Vivado estimate, vectorless)

| Component | Power (W) |
|---|---:|
| **Total on-chip** | **2.869** |
| Processing system (ARM PS8) | 2.306 |
| Programmable logic dynamic (clocks + logic + signals + BRAM) | 0.241 |
| Device static | 0.322 |

Confidence level reported by Vivado: **Medium**. No simulation activity file was provided, so switching activity is assumed. Most of the estimated power is the ARM processing system, not the FINN logic.

## Interpretation

- **No DSPs used.** With 1-bit weights and activations, each multiply-accumulate becomes XNOR + popcount, which maps to LUTs instead of DSP multipliers. That's the main reason binarized networks fit small FPGAs.
- **The folding is balanced.** Using `cycles ≈ (MH/PE) × (MW/SIMD)`, the four MVAUs need 4 × 16 = 64, 8 × 8 = 64, 8 × 8 = 64 and 1 × 8 = 8 cycles per image. With no single bottleneck, the analytical compute-pipeline rate at 100 MHz is about 100 MHz / 64 ≈ **1.56 M images/s**. This is an **analytical estimate** of the dataflow pipeline only. It ignores DMA and host overheads and was **not measured**.
- **The shell is about half the design.** Roughly 9.4k of the 17.7k LUTs are DMA and interconnect logic, not the network.
- **The biggest layer dominates.** MVAU 0 (784 inputs) uses about half the accelerator's LUTs and all 11 of its block RAMs.

## Not done / limitations

- Not run on hardware: no board yet (AUP-ZU3 or KV260 planned). The bitstream (`resizer.bit`) and PYNQ driver are packaged and ready for when one is available.
- Accuracy on the board wasn't measured. The tutorial's software verification is separate (`tfc_end2end_verification.ipynb`).
- Single implementation run at 100 MHz; no Fmax sweep.

## Raw reports

In [`../results/tfc_w1a1_aupzu3/`](../results/tfc_w1a1_aupzu3/): `top_wrapper_utilization_placed.rpt`, `top_wrapper_timing_summary_routed.rpt`, `top_wrapper_power_routed.rpt`, `top_wrapper_route_status.rpt`, `synth_report.xml` (hierarchical post-route utilization).
