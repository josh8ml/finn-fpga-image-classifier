# FINN setup log

Every command, version, error and fix from setting up the [FINN](https://github.com/Xilinx/finn) compiler.
Only results that actually ran are marked PASSED.

- **Part A:** main machine, Windows + WSL2, official Docker flow (2026-10-14/15)
- **Part B:** earlier dry run on a small Linux VM without Docker access (2026-10-07)

---

## Part A: Windows + WSL2 (official Docker flow)

| Item | Value |
|---|---|
| Host | Windows PC, CPU virtualization enabled |
| Linux | WSL2, Ubuntu 26.04.1 LTS |
| Memory / disk | 15 GB available to WSL; ~1.4 TB free |
| Docker | Docker Engine (installed in WSL via `get.docker.com`), runs without sudo |
| FINN | `main` @ `38c42bf`, tag `v1.0.0-alpha` |
| Vivado / Vitis / Vitis HLS | 2024.2 in `/tools/Xilinx` (device families: Zynq UltraScale+ MPSoC, Kria SOMs, Zynq-7000, Artix-7) |

### Commands

```powershell
# PowerShell (admin)
wsl --update
wsl --install -d Ubuntu          # "-d Ubuntu-22.04" returned "invalid distribution name"
```
```bash
# Ubuntu (WSL)
sudo apt-get update && sudo apt-get install -y curl git
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker $USER    # then: exit; wsl --shutdown; wsl
docker run --rm hello-world
cd ~ && git clone https://github.com/Xilinx/finn.git && cd finn
./run-docker.sh quicktest
./run-docker.sh notebook
```
```bash
# Vivado/Vitis 2024.2: installer run in a temporary ubuntu:22.04 container (details in windows-wsl-setup.md)
echo 'export FINN_XILINX_PATH=/tools/Xilinx' >> ~/.bashrc
echo 'export FINN_XILINX_VERSION=2024.2' >> ~/.bashrc
source ~/.bashrc
cd ~/finn && ./run-docker.sh                 # FINN shell with Vivado mounted
echo $XILINX_VIVADO                          # -> /tools/Xilinx/Vivado/2024.2
pytest -v "tests/fpgadataflow/test_fpgadataflow_fmpadding.py::test_fpgadataflow_fmpadding[cppsim-idt0-1-2-pad0-idim0]" \
          "tests/fpgadataflow/test_fpgadataflow_fmpadding.py::test_fpgadataflow_fmpadding[rtlsim-idt0-1-2-pad0-idim0]"
pytest -v tests/fpgadataflow/test_fpgadataflow_checksum.py::test_fpgadataflow_checksum
```

### Results

| Step | Result |
|---|---|
| `docker run --rm hello-world` | PASSED |
| `./run-docker.sh quicktest` (FINN's non-Vivado test suite) | PASSED: **2736 passed, 16 skipped, 5 xfailed, 1 xpassed, 1449 warnings in 72 s** |
| Notebook `basics/0_how_to_work_with_onnx` | Ran in Jupyter, no errors |
| Notebook `basics/1_brevitas_network_import_via_QONNX` | First attempt failed at cell 8 (see problems). Run on its own: PASSED; Brevitas vs. QONNX vs. FINN-ONNX outputs match (`np.isclose(...).all()` → `True`, `True`) |
| Vivado 2024.2 install | DONE: `/tools/Xilinx/{Vivado,Vitis,Vitis_HLS}/2024.2` present |
| `test_fpgadataflow_fmpadding` (cppsim + rtlsim, part `xczu3eg`) | PASSED, 2/2 in 2.2 s. The layer is specialized to **RTL** (`FMPadding_rtl`), so this exercises Vivado's simulator (XSI) but **not** Vitis HLS; rtlsim cycle count matched FINN's estimate (test asserts within 10 cycles) |
| `test_fpgadataflow_checksum` (part `xczu3eg`, `MVAU_hls` + `CheckSum_hls`) | PASSED in 69.9 s: **Vitis HLS synthesis + cppsim + rtlsim work** |
| `./run-docker.sh quicktest verify` | NOT RUN: its test targets a Versal part (`xcvc1902`), which wasn't installed |
| Full bitstream / out-of-context synthesis | NOT RUN yet |

### Problems and fixes

| Problem | Cause | Fix |
|---|---|---|
| PowerShell `VirtualizationFirmwareEnabled` reported False | That check can be misleading | Task Manager → Performance → CPU showed *Virtualization: Enabled* |
| `wsl --install -d Ubuntu-22.04` → "invalid distribution name" | Name not in WSL's online distro list | `wsl --update`, then `wsl --install -d Ubuntu` (installed 26.04.1) |
| Linux commands typed into PowerShell | Two different shells | `wsl` then `cd ~`; Linux prompt ends in `$` |
| `df -h ~` reported 954 GB free | That's the max size of WSL's virtual disk | Real free space: `df -h /mnt/c` |
| Notebook 1, cell 8: `OSError: [Errno 98] Address already in use` | Notebook 0's kernel still held Netron's port 8081. `showInNetron()` retries after `netron.stop()`, but that only stops servers in its own process | Shut down notebook 0's kernel (or restart the Jupyter server), rerun notebook 1 |
| Vivado installer on Ubuntu 26.04 | Newer than the installer supports | Ran the installer inside a temporary `ubuntu:22.04` container, installing to a mounted `/tools/Xilinx` |
| `quicktest verify` unsuitable | Its HLS LayerNorm test uses Versal part `xcvc1902`; only Zynq/Kria/Artix families installed | Verified with FINN tests that target `xczu3eg` instead (above) |
| `pytest -k "... idim0 and simd0 ..."` → `32339 deselected` | Guessed parametrize IDs wrong (actual: `[rtlsim-idt0-1-2-pad0-idim0]`) | `pytest --collect-only -q <file>` to list exact IDs |

### Notes

- Ubuntu 26.04 is newer than what Vivado 2024.2 officially supports. FINN runs Vivado inside its own Ubuntu 22.04 container, so the Vivado installer is run inside a temporary `ubuntu:22.04` container too (see `windows-wsl-setup.md`).

---

## Part B: Linux VM dry run (no Docker access)

A first pass on a small Linux VM (2 vCPU, 7 GB RAM, Ubuntu 24.04, Python 3.13) to find setup problems early. Its network blocked Docker Hub, `download.pytorch.org` and `download.amd.com`, so the Docker image could not be built. FINN's documented **local (no-Docker) install** was used instead.

```bash
git clone https://github.com/Xilinx/finn.git && cd finn       # main @ 38c42bf
pip install --break-system-packages uv && uv python install 3.10
sudo bash scripts/install-system-deps.sh
FINN_PYTHON=$(uv python find 3.10) ./setup-local.sh --skip-xsi
source scripts/finn-env.sh
./scripts/quicktest-local.sh
jupyter nbconvert --to notebook --execute notebooks/basics/0_how_to_work_with_onnx.ipynb
jupyter nbconvert --to notebook --execute notebooks/basics/1_brevitas_network_import_via_QONNX.ipynb
```

| Step | Result |
|---|---|
| `setup-local.sh --skip-xsi` | PASSED (~14 min, "All imports successful") |
| `quicktest-local.sh` | PASSED: imports; streamline tests 9/9; utility tests 13/13 |
| Both basics notebooks (each run as a separate process) | PASSED, 0 error cells |

Versions: torch 2.8.0 (PyPI build, since the cu126 index was blocked), brevitas 0.12.2.dev51 (`aad4d5a2`), qonnx `f5c9819`, onnx 1.17.0, onnxruntime 1.18.1, numpy 1.24.1, finn-hlslib `8d979e2`.

| Problem | Fix |
|---|---|
| System Python 3.13 too new for FINN's pinned deps (numpy 1.24.1) | Python 3.10 via `uv`, passed with `FINN_PYTHON` |
| Docker image can't be built on that VM | Used the documented local install |

---

## Upstream observations (possible contributions, to be checked against open issues)

1. `setup-local.sh` and `run-docker.sh` still suggest `FINN_XILINX_VERSION=2022.2` in help/warning text (and `docker/finn_entrypoint.sh` says "Our recommendation is to use version 2022.2" for Vitis HLS), while `docs/finn/getting_started.rst` requires 2024.2. Confirmed on `dev` @ `e9dd2eb` (2026-10-05) for the first two.
2. The basics notebooks can't be run one after another in the same Jupyter session: notebook 1's first `showInNetron()` fails with `Address already in use` while notebook 0's kernel is alive. Reproduced on the WSL setup.
3. `README.md` says only Docker-based execution is supported, while `getting_started.rst` documents a local install.
4. `getting_started.rst` tells users to verify the install with `./run-docker.sh quicktest verify`, but that check builds for a Versal part (`xcvc1902`); users who install only Zynq/Kria device families (a common way to save disk) can't use it, and this isn't mentioned.
5. `src/finn/builder/build_dataflow_checks.py`: the Vivado-version error messages use `f"found {v[0]}.{v[1] if v else 'unknown'}"`. The guard only covers `v[1]`, so when Vivado isn't detected (`get_vivado_version()` returns `None`), building the message raises `TypeError: 'NoneType' object is not subscriptable` instead of reporting the problem. Hit with an estimate-only build using `board="AUP-ZU3_8GB"` and no Vivado; the same pattern affects the V80, SLASH and MLO checks. Present on `dev` @ `98306537d` (2026-10-27). The existing `tests/util/test_build_dataflow_checks.py` covers wrong Vivado versions but not a missing one.
