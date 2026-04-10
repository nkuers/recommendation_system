import argparse
import itertools
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_GRID = ROOT / "scripts" / "run_grid.py"


def _parse_list(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def _run(cmd, dry_run=False):
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
    return proc.returncode


def _base_cmd(args, output_path, tag, variant):
    cmd = [
        sys.executable,
        str(RUN_GRID),
        "--models",
        "timeweaver",
        "--datasets",
        args.datasets,
        "--seeds",
        args.seeds,
        "--device",
        args.device,
        "--timeweaver-variant",
        variant,
        "--output",
        str(output_path),
        "--tag",
        tag,
    ]
    if args.epochs_override is not None:
        cmd.extend(["--epochs-override", str(args.epochs_override)])
    if args.skip_if_done:
        cmd.append("--skip-if-done")
    return cmd


def _exp_a(args):
    out = args.output_dir / "timeweaver_param_A_contrastive.json"
    weights = _parse_list(args.a_weights, float)
    temps = _parse_list(args.a_temps, float)
    for w, t in itertools.product(weights, temps):
        tag = f"A_w{w}_t{t}"
        cmd = _base_cmd(args, out, tag=tag, variant=args.aug_variant)
        cmd.extend(["--tw-contrastive-weight", str(w), "--tw-contrastive-temp", str(t)])
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def _exp_b2(args):
    out = args.output_dir / "timeweaver_param_B2_aug_strength_prob.json"
    strengths = _parse_list(args.b2_strengths, float)
    probs = _parse_list(args.b2_probs, float)
    for s, p in itertools.product(strengths, probs):
        tag = f"B2_s{s}_p{p}"
        cmd = _base_cmd(args, out, tag=tag, variant=args.aug_variant)
        cmd.extend(["--tw-aug-strength", str(s), "--tw-aug-prob", str(p)])
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def _exp_b1(args):
    out = args.output_dir / "timeweaver_param_B1_warmup_ramp.json"
    warmups = _parse_list(args.b1_warmups, int)
    ramps = _parse_list(args.b1_ramps, int)
    for w, r in itertools.product(warmups, ramps):
        tag = f"B1_warm{w}_ramp{r}"
        cmd = _base_cmd(args, out, tag=tag, variant=args.aug_variant)
        cmd.extend(["--tw-time-aug-warmup", str(w), "--tw-time-aug-ramp", str(r)])
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def _exp_mode(args):
    out = args.output_dir / "timeweaver_param_mode_aug.json"
    modes = _parse_list(args.modes, str)
    for mode in modes:
        tag = f"MODE_{mode}"
        cmd = _base_cmd(args, out, tag=tag, variant=args.aug_variant)
        cmd.extend(["--tw-time-aug-mode", mode])
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def _exp_c(args):
    out = args.output_dir / "timeweaver_param_C_gate_hidden.json"
    hiddens = _parse_list(args.c_gate_hiddens, int)
    for h in hiddens:
        tag = f"C_gate_hidden{h}"
        cmd = _base_cmd(args, out, tag=tag, variant=args.gate_variant)
        cmd.extend(["--tw-gate-hidden", str(h)])
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run prioritized TimeWeaver parameter sweeps.")
    parser.add_argument("--datasets", required=True, help="Comma-separated datasets for run_grid.")
    parser.add_argument("--seeds", default="42,2023,2024")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--output-dir", default="results/param_sweeps")
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--experiments",
        default="A,B2,B1,MODE,C",
        help="Subset from A,B2,B1,MODE,C (comma-separated).",
    )

    # Variant selection
    parser.add_argument("--aug-variant", default="aug", choices=["aug_only", "aug"])
    parser.add_argument("--gate-variant", default="gate", choices=["gate_discrete", "gate"])

    # A: contrastive
    parser.add_argument("--a-weights", default="0,0.01,0.03,0.05,0.1")
    parser.add_argument("--a-temps", default="0.1,0.2,0.5")

    # B2: augmentation strength/prob
    parser.add_argument("--b2-strengths", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--b2-probs", default="0.05,0.1,0.15,0.3")

    # B1: schedule
    parser.add_argument("--b1-warmups", default="0,500,1000,2000")
    parser.add_argument("--b1-ramps", default="500,1000,2000,4000")

    # MODE
    parser.add_argument("--modes", default="mix,median,jitter")

    # C: gate hidden
    parser.add_argument("--c-gate-hiddens", default="8,16,32,64")

    args = parser.parse_args()
    args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
    mapping = {
        "A": _exp_a,
        "B2": _exp_b2,
        "B1": _exp_b1,
        "MODE": _exp_mode,
        "C": _exp_c,
    }
    for name in plan:
        if name not in mapping:
            raise SystemExit(f"Unknown experiment group: {name}")
        print(f"\n[RUN GROUP] {name}")
        rc = mapping[name](args)
        if rc != 0:
            raise SystemExit(rc)

    print("\n[DONE] Parameter sweeps finished.")
    print(f"[OUT] {args.output_dir}")


if __name__ == "__main__":
    main()
