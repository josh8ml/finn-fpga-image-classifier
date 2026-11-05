"""Throughput/area trade-off: run FINN estimate-only builds for several target FPS values.

    python finn_build/sweep.py --model w4a4 --targets 1000 2000 5000 10000 20000 50000

For each target, FINN's automatic folding (step_target_fps_parallelization) picks
PE/SIMD per layer. This records the resulting analytical estimates in
runs/<model>/folding_sweep.json. Result type: estimate (FINN analytical model).
No Vivado needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def summarize(report_dir: Path) -> dict:
    perf = json.loads((report_dir / "estimate_network_performance.json").read_text())
    res = json.loads((report_dir / "estimate_layer_resources.json").read_text())
    layers = {k: v for k, v in res.items() if k != "total"}
    return {
        "estimated_throughput_fps": round(perf["estimated_throughput_fps"], 1),
        "bottleneck_layer": perf["max_cycles_node_name"],
        "bottleneck_cycles": perf["max_cycles"],
        "estimated_latency_us": round(perf["estimated_latency_ns"] / 1000, 1),
        "LUT": round(sum(v.get("LUT", 0) for v in layers.values())),
        "BRAM_18K": sum(v.get("BRAM_18K", 0) for v in layers.values()),
        "URAM": sum(v.get("URAM", 0) for v in layers.values()),
        "DSP": sum(v.get("DSP", 0) for v in layers.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="w4a4")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--targets", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000, 50000])
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "runs" / args.model
    rows = []
    for t in args.targets:
        out = run_dir / "sweep" / f"fps_{t}"
        cmd = [sys.executable, str(REPO / "finn_build" / "build.py"), "--model", args.model,
               "--mode", "estimate", "--run-dir", str(run_dir), "--out-dir", str(out), "--target-fps", str(t)]
        print(f"target {t} fps ...", flush=True)
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            print(f"  build failed (see {out}/build_dataflow.log)")
            continue
        row = {"target_fps": t, **summarize(out / "report")}
        rows.append(row)
        print(f"  est {row['estimated_throughput_fps']:.0f} fps, {row['LUT']} LUT, {row['BRAM_18K']} BRAM18, "
              f"{row['DSP']} DSP, bottleneck {row['bottleneck_layer']}")

    result = {"result_type": "estimate (FINN analytical model)", "model": args.model,
              "part": "xczu3eg-sfvc784-2-e (AUP-ZU3)", "clock_mhz": 100, "rows": rows}
    (run_dir / "folding_sweep.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {run_dir / 'folding_sweep.json'}")


if __name__ == "__main__":
    main()
