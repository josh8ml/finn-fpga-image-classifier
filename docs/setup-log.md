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
| Vivado / Vitis | 2024.2, install in progress (not used for any result below) |

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

### Results

| Step | Result |
|---|---|
| `docker run --rm hello-world` | PASSED |
| `./run-docker.sh quicktest` (FINN's non-Vivado test suite) | PASSED: **2736 passed, 16 skipped, 5 xfailed, 1 xpassed, 1449 warnings in 72 s** |
| Notebook `basics/0_how_to_work_with_onnx` | Ran in Jupyter, no errors |
| Notebook `basics/1_brevitas_network_import_via_QONNX` | First attempt failed at cell 8 (see problems). Run on its own: PASSED; Brevitas vs. QONNX vs. FINN-ONNX outputs match (`np.isclose(...).all()` → `True`, `True`) |
| Vivado-dependent steps (HLS, cppsim, rtlsim, synthesis) | NOT RUN yet |

### Problems and fixes

| Problem | Cause | Fix |
|---|---|---|
| PowerShell `VirtualizationFirmwareEnabled` reported False | That check can be misleading | Task Manager → Performance → CPU showed *Virtualization: Enabled* |
| `wsl --install -d Ubuntu-22.04` → "invalid distribution name" | Name not in WSL's online distro list | `wsl --update`, then `wsl --install -d Ubuntu` (installed 26.04.1) |
| Linux commands typed into PowerShell | Two different shells | `wsl` then `cd ~`; Linux prompt ends in `$` |
| `df -h ~` reported 954 GB free | That's the max size of WSL's virtual disk | Real free space: `df -h /mnt/c` |
| Notebook 1, cell 8: `OSError: [Errno 98] Address already in use` | Notebook 0's kernel still held Netron's port 8081. `showInNetron()` retries after `netron.stop()`, but that only stops servers in its own process | Shut down notebook 0's kernel (or restart the Jupyter server), rerun notebook 1 |

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

1. `setup-local.sh` and `run-docker.sh` still suggest `FINN_XILINX_VERSION=2022.2` in help/warning text, while `docs/finn/getting_started.rst` requires 2024.2. Confirmed on `dev` @ `e9dd2eb` (2026-10-05).
2. The basics notebooks can't be run one after another in the same Jupyter session: notebook 1's first `showInNetron()` fails with `Address already in use` while notebook 0's kernel is alive. Reproduced on the WSL setup.
3. `README.md` says only Docker-based execution is supported, while `getting_started.rst` documents a local install.
