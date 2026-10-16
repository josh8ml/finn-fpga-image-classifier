# PROJECT_NOTES — Learning FINN

Working notes for my FINN project: environment, setup, how FINN works, and what actually ran.
**Rule for this file:** anything listed under *Results* actually ran. Anything not yet run is marked **NOT RUN**.

---

## 1. Environment

| Item | Value |
|---|---|
| Host | Windows PC, CPU virtualization enabled |
| Linux | WSL2 → Ubuntu 26.04.1 LTS |
| Memory / disk | 15 GB available to WSL, 1.4 TB free on C: |
| Container runtime | Docker Engine (inside WSL), runs without `sudo` |
| FINN | `Xilinx/finn` `main` @ `38c42bf`, tag `v1.0.0-alpha` (2026-09-06) |
| FINN container | Ubuntu 22.04 + Python 3.10 + PyTorch 2.8 + Brevitas + QONNX (built by `run-docker.sh`) |
| Vivado / Vitis / Vitis HLS | 2024.2 in `/tools/Xilinx` (Zynq UltraScale+, Kria, Zynq-7000, Artix-7 device families) |
| FPGA board | none yet (target: AUP-ZU3 or Kria KV260) |

## 2. Setup steps (summary)

1. Confirmed virtualization is enabled (Task Manager → Performance → CPU).
2. Installed WSL2 + Ubuntu (`wsl --update`, `wsl --install -d Ubuntu`).
3. Installed Docker Engine in Ubuntu (`get.docker.com`), added my user to the `docker` group.
4. Cloned FINN and ran the official non-Vivado test suite in Docker (`./run-docker.sh quicktest`).
5. Ran the two "basics" tutorial notebooks (`./run-docker.sh notebook`).
6. Installed Vivado/Vitis 2024.2 (installer run in a temporary Ubuntu 22.04 container), pointed FINN at it with `FINN_XILINX_PATH` / `FINN_XILINX_VERSION`, and verified RTL simulation and Vitis HLS with FINN's own tests.

Full command-by-command log with versions and errors: `setup-log.md`.

## 3. What FINN does (one paragraph)

FINN is a compiler from AMD Research that turns a **quantized neural network** (weights and activations of a few bits) into a **custom FPGA accelerator**. FINN doesn't run the network on a general-purpose processor. It generates a **streaming dataflow architecture**: every layer gets its own dedicated hardware block, and the blocks are chained with FIFOs, so all layers work at the same time on different images, like a CPU pipeline. Because each layer is a separate block, its parallelism can be sized individually. That is the main tuning knob between speed and FPGA area.

## 4. How a quantized network is represented

| Stage | Format | What it looks like |
|---|---|---|
| Training | **Brevitas** (PyTorch library) | Normal PyTorch model using `QuantLinear`, `QuantConv2d`, `QuantReLU`… with bit widths as parameters. Training is quantization-aware: the network learns weights that still work after rounding to e.g. 2 bits. |
| Export | **QONNX** | ONNX graph where quantization is explicit: `Quant` / `BipolarQuant` nodes carry scale, zero-point and bit width. |
| Inside FINN | **FINN-ONNX** | Same ONNX graph, but each tensor carries a **DataType annotation** (e.g. `INT4`, `UINT2`, `BIPOLAR`). Values are still stored as float32 "containers"; the annotation says what integer range they really hold. |

The tensors stay float32 internally, so the same graph can be **executed in Python** (`finn.core.onnx_exec`) at every step. That's how FINN checks each transformation preserves the network's output. Notebook 1 does exactly this: Brevitas, QONNX and FINN-ONNX outputs all match (`np.isclose(...).all() == True`).

## 5. From model to dataflow hardware (the FINN flow)

```
 ┌──────────────── SOFTWARE (runs in Docker, no Vivado needed) ────────────────┐
 │                                                                             │
 │  PyTorch + Brevitas ──export──► QONNX ──► FINN-ONNX                         │
 │  (train, quantize)                         │                                │
 │                                            ▼                                │
 │                               Tidy-up (shapes, datatypes, names)            │
 │                                            ▼                                │
 │                               Streamline: fold scales/biases/BN into        │
 │                               integer MultiThreshold nodes                  │
 │                                            ▼                                │
 │                               Convert to HW layers (MVAU, SWG, Pool, …)     │
 │                               Dataflow partition │ Specialize: HLS or RTL   │
 │                                            ▼                                │
 │                               Folding: choose PE / SIMD per layer           │
 │                                            ▼                                │
 │                       ESTIMATE REPORTS (analytical: cycles, LUT, BRAM, DSP) │
 └────────────────────────────────────────────┬────────────────────────────────┘
                                              ▼
 ┌───────────── GENERATED HARDWARE (needs Vitis HLS / Vivado) ─────────────────┐
 │  HLS layers  → C++ from finn-hlslib → Vitis HLS → Verilog IP                │
 │  RTL layers  → Verilog templates from finn-rtllib                           │
 │  cppsim (C++ sim) · node-by-node rtlsim · stitched-IP rtlsim (cycle-accurate)│
 │  CreateStitchedIP → one Vivado IP block with AXI-Stream in/out              │
 └────────────────────────────────────────────┬────────────────────────────────┘
                                              ▼
 ┌──────────────── FPGA-SPECIFIC (needs Vivado + a target part/board) ─────────┐
 │  Out-of-context synthesis → real LUT/FF/BRAM/DSP + achievable clock (Fmax)  │
 │  Zynq shell: DMA + PS block design → bitstream + Python (PYNQ) driver       │
 │  Run on the board → measured throughput / latency / accuracy                │
 └─────────────────────────────────────────────────────────────────────────────┘
```

**Key steps explained**
- **Streamlining.** The trained net has floating-point scales, biases and batch-norm everywhere. FINN moves and merges them, and replaces "scale → add → quantized activation" with a single **MultiThreshold** node: compare the integer accumulator against a few precomputed thresholds. After this, everything in the datapath is integer math, which is cheap in FPGA logic.
- **HW layers.** Fully-connected and conv layers become **MVAU** (Matrix-Vector-Activation Unit). Conv layers also get a **ConvolutionInputGenerator** (sliding window, built from line buffers). Each one is a streaming hardware block.
- **HLS vs RTL.** Each HW layer is implemented either as C++ from `finn-hlslib`, compiled to Verilog by **Vitis HLS**, or as hand-written Verilog from `finn-rtllib`. FINN picks per layer.
- **Folding (PE, SIMD).** An MVAU computes an `MH × MW` matrix-vector product. **SIMD** = inputs processed per cycle, **PE** = outputs processed per cycle (constraints: `MW % SIMD == 0`, `MH % PE == 0`). Cycles per vector ≈ `(MH/PE) × (MW/SIMD)`. PE = SIMD = 1 is the smallest and slowest; full parallelism is the fastest and largest.

## 6. Roles of the tools

| Tool | Role | Needed for |
|---|---|---|
| **Brevitas** | Quantization-aware training in PyTorch; exports QONNX | Creating the model |
| **ONNX / QONNX** | The graph format FINN reads and rewrites, pass by pass | Whole flow |
| **FINN compiler** | Python transformations: streamline, convert, fold, estimate, generate code | Whole flow |
| **Vitis HLS** | Compiles C++ layer templates into Verilog IP | HLS layers, cppsim |
| **RTL simulation (XSI / Vivado simulator)** | Cycle-accurate simulation of generated Verilog → real cycle counts, functional check | rtlsim results |
| **Vivado** | Stitches IP, synthesizes / places / routes for a specific part; produces bitstream and utilization and timing reports | Real resources, Fmax, bitstream |
| **PYNQ** (on board) | Linux + Python on Zynq; loads bitstream, runs the FINN-generated driver | On-board measurements |

## 7. How the metrics relate

FINN's estimate formulas (`step_generate_estimate_reports`, `dataflow_performance` analysis):

- **Throughput (images/s)** ≈ `f_clk / max_cycles`. `max_cycles` is the cycle count of the **slowest layer**. Like a pipeline, the slowest stage sets the rate.
- **Latency (s)** ≈ `critical_path_cycles / f_clk`. That's the sum of per-layer cycles along the path; FINN's docs call this estimate pessimistic. Stitched-IP rtlsim gives the real cycle count.
- **Raising PE/SIMD** on the bottleneck layer reduces its cycles and raises throughput, but costs more **LUTs** (adders and compare logic), and **DSPs** for wider multiplies. Raising PE/SIMD on *other* layers wastes area without improving throughput. The goal is a **balanced pipeline**.
- **Weights** live on-chip: in **BRAM/URAM**, or in LUTRAM for small layers. More parameters × more bits means more memory. Higher PE/SIMD also means wider memory ports, which can use more BRAMs even with the same number of weights.
- **Bit width** is the biggest lever. Fewer bits means smaller multipliers (at 1–4 bits they're built from LUTs, not DSPs), smaller weight memories and fewer thresholds, but usually lower **accuracy**.
- **Clock frequency** is set as a target (`synth_clk_period_ns`). Whether it's achieved is only known after Vivado place-and-route (timing report). Throughput estimates assume the target clock.
- **Power** grows with logic used, toggle rate and clock frequency. It's only meaningful from Vivado's power report (estimated) or a measurement on the board. FINN's estimates don't model power.

**What counts as measured, simulated or estimated in this project**

| Number | Source | Label |
|---|---|---|
| Accuracy (software) | Python / ONNX execution | Measured (software) |
| Cycles, LUT, BRAM, DSP from estimate reports | FINN analytical model | **Estimate** |
| Cycles / latency from stitched rtlsim | Vivado simulator | **Simulation** |
| LUT / FF / BRAM / DSP, Fmax after synthesis | Vivado | **Synthesis report** (pre-board) |
| Images/s, latency on board | PYNQ driver on FPGA | **Measured (hardware)** |
| Power | Vivado power report | **Estimate** unless measured on the board |

## 8. Commands used

```bash
# Windows PowerShell (admin)
wsl --update
wsl --install -d Ubuntu

# Ubuntu (WSL)
sudo apt-get update && sudo apt-get install -y curl git
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker $USER          # then: exit → wsl --shutdown → wsl
docker run --rm hello-world
cd ~ && git clone https://github.com/Xilinx/finn.git && cd finn
./run-docker.sh quicktest              # FINN's non-Vivado test suite
./run-docker.sh notebook               # Jupyter at http://127.0.0.1:8888/...
```

## 9. Problems and solutions

| Problem | Cause | Solution |
|---|---|---|
| `quicktest verify` can't be used | Its test targets a Versal part that wasn't installed | Verified with `xczu3eg`-targeted FINN tests (fmpadding, checksum) |
| `wsl --install -d Ubuntu-22.04` → "invalid distribution name" | Name not in WSL's online list | `wsl --update`, then `wsl --install -d Ubuntu` |
| Ran Linux commands in PowerShell | Two different shells | `wsl` → `cd ~`; Linux prompt ends in `$` |
| `df -h ~` showed 954 GB free | That's WSL's virtual-disk max size | Real free space: `df -h /mnt/c` |
| Notebook 1, cell 8: `OSError: [Errno 98] Address already in use` | Netron viewer (port 8081) still held by notebook 0's kernel; `showInNetron`'s retry calls `netron.stop()`, which only stops servers in its *own* process | Shut down notebook 0's kernel (or restart the server), rerun notebook 1. **Candidate upstream fix (Phase 5).** |

## 10. Results from the official tutorial and tests

| What | Result | Type |
|---|---|---|
| `./run-docker.sh quicktest` | **2736 passed, 16 skipped, 5 xfailed, 1 xpassed** (72 s) | Software tests |
| `basics/0_how_to_work_with_onnx` | Ran (built a small ONNX graph of Add nodes, replaced two consecutive Adds with one Sum node; output matches the NumPy reference before and after) | Software |
| `basics/1_brevitas_network_import_via_QONNX` | Pretrained quantized MNIST MLP: Brevitas vs. QONNX vs. FINN-ONNX outputs equal (`True`, `True`) | Software |
| `test_fpgadataflow_fmpadding` cppsim + rtlsim (`xczu3eg`, RTL layer) | 2 passed (2.2 s); rtlsim cycles within 10 of FINN's estimate | Simulation (Vivado XSI) |
| `test_fpgadataflow_checksum` (`xczu3eg`, `MVAU_hls` + `CheckSum_hls`) | 1 passed (69.9 s): Vitis HLS synthesis + cppsim + rtlsim | HLS + simulation |
| FPGA resources / Fmax / power | **NOT RUN** | — |
| On-board throughput | **NOT RUN** (no board) | — |

## 11. Next steps

- Work through `end2end_example/bnn-pynq/tfc_end2end_example.ipynb`. It's the same MNIST network as notebook 1, taken all the way to hardware.
- Phase 4: own image-classifier project (software baseline vs. FINN estimate/sim/synthesis).
