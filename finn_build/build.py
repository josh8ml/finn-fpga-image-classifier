"""FINN build for the Fashion-MNIST quantized CNN, targeting the AUP-ZU3 (xczu3eg).

Run inside the FINN Docker container, from the repo root:

    # analytical estimates only: no Vivado needed, ~1 minute
    python finn_build/build.py --model w4a4 --mode estimate

    # full flow: HLS + RTL simulation + Vivado synthesis/place/route + bitstream
    python finn_build/build.py --model w4a4 --mode full

Output goes to runs/<model>/finn_<mode>/ (reports in its report/ subfolder).

Two custom steps are added before FINN's own flow, following FINN's
4_advanced_builder_settings tutorial:
  * pre-processing: divide by 255 inside the graph, so the accelerator takes
    raw 8-bit pixels (UINT8) instead of floats;
  * post-processing: a top-1 node, so the accelerator outputs the class index.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BOARD = "AUP-ZU3_8GB"
CLOCK_NS = 10.0  # 100 MHz


def custom_step_add_pre_proc(model, cfg):
    import torch
    from brevitas.export import export_qonnx
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.merge_onnx_models import MergeONNXModels

    from finn.util.pytorch import ToTensor

    ishape = model.get_tensor_shape(model.graph.input[0].name)
    preproc_path = cfg.output_dir + "/preproc.onnx"
    export_qonnx(ToTensor(), torch.randn(ishape), preproc_path, opset_version=11)
    preproc = ModelWrapper(preproc_path)
    preproc.set_tensor_datatype(preproc.graph.input[0].name, DataType["UINT8"])
    return model.transform(MergeONNXModels(preproc))


def custom_step_add_post_proc(model, cfg):
    from qonnx.transformation.insert_topk import InsertTopK

    return model.transform(InsertTopK(k=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="w4a4")
    p.add_argument("--mode", choices=["estimate", "full"], default="estimate")
    p.add_argument("--target-fps", type=int, default=10000, help="FINN picks PE/SIMD to reach this")
    p.add_argument("--run-dir", default=None, help="default runs/<model>")
    p.add_argument("--out-dir", default=None, help="default runs/<model>/finn_<mode>")
    args = p.parse_args()

    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as build_cfg

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "runs" / args.model
    model_file = run_dir / "model_qonnx.onnx"
    if not model_file.exists():
        sys.exit(f"{model_file} not found: run scripts/export_qonnx.py --model {args.model} first")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / f"finn_{args.mode}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    verify = [
        build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,
    ]
    common = dict(
        output_dir=str(out_dir),
        target_fps=args.target_fps,
        mvau_wwidth_max=80,
        synth_clk_period_ns=CLOCK_NS,
        inject_steps_before={
            "step_qonnx_to_finn": [custom_step_add_pre_proc, custom_step_add_post_proc]
        },
        verify_input_npy=str(run_dir / "verify_input.npy"),
        verify_expected_output_npy=str(run_dir / "verify_expected.npy"),
        save_intermediate_models=True,
    )

    if args.mode == "estimate":
        # Estimates only need the FPGA part, not the board shell. Passing the part
        # directly also keeps this mode usable on machines without Vivado.
        from finn.util.basic import part_map

        cfg = build.DataflowBuildConfig(
            fpga_part=part_map[BOARD],
            steps=build_cfg.estimate_only_dataflow_steps,
            generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
            verify_steps=verify,
            **common,
        )
    else:
        cfg = build.DataflowBuildConfig(
            board=BOARD,
            steps=build_cfg.default_build_dataflow_steps,
            shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
            default_swg_exception=True,
            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                build_cfg.DataflowOutputType.BITFILE,
                build_cfg.DataflowOutputType.PYNQ_DRIVER,
                build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
            ],
            verify_steps=verify
            + [
                build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
                build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
            ],
            **common,
        )

    status = build.build_dataflow_cfg(str(model_file), cfg)
    print(f"FINN build ({args.mode}) finished with status {status}. Outputs: {out_dir}")
    sys.exit(status)


if __name__ == "__main__":
    main()
