#!/bin/bash
# Run the whole software + estimate pipeline. Run INSIDE the FINN Docker container,
# from the repo root:   bash scripts/run_all.sh
#
# Steps: tests -> train FP32/W4A4/W2A2 -> CPU baseline -> export to QONNX ->
#        FINN estimate builds -> FINN accuracy check -> throughput/area sweep -> summary.
# Needs no Vivado. The full hardware build is separate:
#        python finn_build/build.py --model w4a4 --mode full
set -euo pipefail
cd "$(dirname "$0")/.."

FINN_ACC_IMAGES="${FINN_ACC_IMAGES:-10000}"   # images for the FINN accuracy check (all = 10000)

step() { echo; echo "=================== $* ==================="; }

step "1/8 Tests"
python -m pytest tests -q -p no:cacheprovider -W ignore

step "2/8 Train FP32 baseline"
python scripts/train.py --model fp32
step "3/8 Train W4A4 and W2A2"
python scripts/train.py --model w4a4
python scripts/train.py --model w2a2

step "4/8 CPU baseline (FP32)"
python scripts/cpu_baseline.py

step "5/8 Export quantized models to QONNX"
python scripts/export_qonnx.py --model w4a4
python scripts/export_qonnx.py --model w2a2

step "6/8 FINN estimate-only builds"
python finn_build/build.py --model w4a4 --mode estimate
python finn_build/build.py --model w2a2 --mode estimate

step "7/8 Accuracy of FINN-transformed models on ${FINN_ACC_IMAGES} test images"
python scripts/finn_accuracy.py --model w4a4 --images "$FINN_ACC_IMAGES"
python scripts/finn_accuracy.py --model w2a2 --images "$FINN_ACC_IMAGES"

step "8/8 Throughput/area sweeps and summary"
python finn_build/sweep.py --model w4a4
python finn_build/sweep.py --model w2a2
python scripts/collect_results.py

echo
echo "All done. Summary: results/summary.md"
