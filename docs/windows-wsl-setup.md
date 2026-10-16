# Running FINN on Windows (WSL2 + Docker)

The steps that worked on my machine. FINN officially targets Linux + Docker; running it under WSL2 isn't officially tested by the FINN team, but everything inside the FINN container is the standard flow.

## 1. Prerequisites

- Hardware virtualization enabled (Task Manager → Performance → CPU → *Virtualization: Enabled*; otherwise enable SVM / VT-x in the BIOS).
- Disk: ~40 GB for FINN + Docker image; Vivado/Vitis 2024.2 needs much more (see section 5).

## 2. WSL2 + Ubuntu

PowerShell as Administrator:

```powershell
wsl --update
wsl --install -d Ubuntu
```

Create the Linux user when Ubuntu first starts. Check with `wsl -l -v` (VERSION should be 2).

## 3. Docker Engine inside Ubuntu

```bash
sudo apt-get update && sudo apt-get install -y curl git
cat /etc/wsl.conf                      # needs [boot] systemd=true; if missing:
printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER          # FINN requires docker without sudo
```

Restart WSL (`exit`, then `wsl --shutdown` in PowerShell, then `wsl`), and check: `docker run --rm hello-world`.

## 4. FINN

Clone into the Linux filesystem (`~`), not under `/mnt/c` (slow, permission issues):

```bash
cd ~ && git clone https://github.com/Xilinx/finn.git && cd finn
./run-docker.sh quicktest      # builds the image on first run, then runs non-Vivado tests
./run-docker.sh notebook       # Jupyter; open the http://127.0.0.1:8888/... link in a Windows browser
```

Red warnings about `FINN_XILINX_PATH` / `PLATFORM_REPO_PATHS` / `SLASHKIT_DEB_PACKAGE` are expected until Vivado is installed.

**Notebook tip:** run one notebook at a time. Shut down the previous notebook's kernel (File → Close and Shut Down Notebook) before starting the next, or the Netron viewer fails with `Address already in use`.

## 5. Vivado/Vitis 2024.2

FINN requires exactly 2024.2. Because my WSL Ubuntu (26.04) is newer than what the installer supports, the installer runs inside a temporary Ubuntu 22.04 container and installs into `/tools/Xilinx` on the WSL filesystem.

1. Download **"AMD Unified Installer for FPGAs & Adaptive SoCs 2024.2: Linux Self Extracting Web Installer"** (~304 MB) from AMD's 2024.2 downloads page (free AMD account + export-control form).
2. In Ubuntu:

```bash
mkdir -p ~/xilinx-installer
cp /mnt/c/Users/*/Downloads/FPGAs_AdaptiveSoCs_Unified_2024.2_*_Lin64.bin ~/xilinx-installer/
sudo mkdir -p /tools/Xilinx
docker run -it --rm -v /tools/Xilinx:/tools/Xilinx -v ~/xilinx-installer:/installer ubuntu:22.04 bash
```

3. Inside the container:

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y locales libtinfo5 libncurses5 libx11-6 libxext6 libxrender1 libxtst6 libxi6 libfreetype6 fontconfig
locale-gen en_US.UTF-8 && export LANG=en_US.UTF-8
cd /installer && chmod +x FPGAs_AdaptiveSoCs_Unified_2024.2_*_Lin64.bin
./FPGAs_AdaptiveSoCs_Unified_2024.2_*_Lin64.bin --keep --noexec --target /installer/extracted
cd /installer/extracted
./xsetup -b AuthTokenGen          # AMD account login
./xsetup -b ConfigGen             # choose "Vitis" → /root/.Xilinx/install_config.txt
```

4. Edit the `Modules=` line in `/root/.Xilinx/install_config.txt` so that only these are `:1`: *Install Devices for Kria SOMs and Starter Kits*, *Zynq UltraScale+ MPSoC*, *Zynq-7000*, *Artix-7* (everything else `:0`). This greatly reduces the download.
5. Install (accepts AMD's EULAs):

```bash
./xsetup --agree XilinxEULA,3rdPartyEULA --batch Install --config /root/.Xilinx/install_config.txt
```

6. Point FINN at it (permanently) and open a FINN shell:

```bash
echo 'export FINN_XILINX_PATH=/tools/Xilinx' >> ~/.bashrc
echo 'export FINN_XILINX_VERSION=2024.2' >> ~/.bashrc
source ~/.bashrc
cd ~/finn && ./run-docker.sh
echo $XILINX_VIVADO     # should print /tools/Xilinx/Vivado/2024.2
```

7. Verify. FINN's own `./run-docker.sh quicktest verify` builds for a **Versal** part, which isn't installed with the device selection above. Instead, run FINN tests that target the AUP-ZU3 part (`xczu3eg`):

```bash
# Vivado simulator via an RTL layer (seconds)
pytest -v "tests/fpgadataflow/test_fpgadataflow_fmpadding.py::test_fpgadataflow_fmpadding[rtlsim-idt0-1-2-pad0-idim0]"
# Vitis HLS synthesis + cppsim + rtlsim (about a minute)
pytest -v tests/fpgadataflow/test_fpgadataflow_checksum.py::test_fpgadataflow_checksum
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `permission denied ... docker.sock` | Group change not applied: `wsl --shutdown`, reopen Ubuntu |
| `Cannot connect to the Docker daemon` | Enable systemd in `/etc/wsl.conf`, restart WSL |
| "invalid distribution name" on `wsl --install` | `wsl --update`; `wsl --list --online` for valid names |
| Very slow builds | Repo is under `/mnt/c`; clone under `~` instead |
| Notebook: `Address already in use` | Shut down other running notebook kernels |
