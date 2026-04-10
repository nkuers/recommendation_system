# scripts/benchmark.py
import argparse
import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.baseline_mfseq import MeanSeqModel
from models.dt_attn import DTAttentionModel
from models.dt_seq import MeanSeqDTModel
from models.hour_seq import MeanSeqHourModel
from models.interest_clock_simple import SimpleInterestClock
from models.lic_v1 import LICv1


DEFAULT_MODELS = [
    "mfseq",
    "dt_attn",
    "dt_seq",
    "hour",
    "interest_clock",
    "lic_v1",
    "lic_v1_decay",
    "lic_v1_timebias",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model efficiency.")
    parser.add_argument("--model", default="all",
                        choices=DEFAULT_MODELS + ["all"],
                        help="Model to benchmark (or 'all').")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--long_window", type=int, default=30)
    parser.add_argument("--short_len", type=int, default=10)
    parser.add_argument("--num_items", type=int, default=10000)
    parser.add_argument("--num_dt_buckets", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=str, default="", help="Output json path.")
    return parser.parse_args()


def build_model(model_type, num_items, num_dt_buckets, emb_dim):
    if model_type == "mfseq":
        return MeanSeqModel(num_items, emb_dim=emb_dim)
    if model_type == "dt_attn":
        return DTAttentionModel(num_items, num_dt_buckets, emb_dim=emb_dim)
    if model_type == "dt_seq":
        return MeanSeqDTModel(num_items, num_dt_buckets, emb_dim=emb_dim)
    if model_type == "hour":
        return MeanSeqHourModel(num_items, emb_dim=emb_dim)
    if model_type == "interest_clock":
        return SimpleInterestClock(num_items, num_dt_buckets, emb_dim=emb_dim)
    if model_type == "lic_v1":
        return LICv1(num_items, num_dt_buckets, emb_dim=emb_dim, use_dt_decay=False)
    if model_type == "lic_v1_decay":
        return LICv1(num_items, num_dt_buckets, emb_dim=emb_dim, use_dt_decay=True)
    if model_type == "lic_v1_timebias":
        return LICv1(num_items, num_dt_buckets, emb_dim=emb_dim, use_time_bias=True)
    raise ValueError(f"Unknown model_type: {model_type}")


def build_inputs(model_type, batch_size, window_size, long_window, short_len,
                 num_items, num_dt_buckets, device):
    items_pad = torch.randint(1, num_items, (batch_size, window_size), device=device)
    mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=device)
    pos = torch.randint(1, num_items, (batch_size,), device=device)
    neg = torch.randint(1, num_items, (batch_size,), device=device)

    if model_type in {"dt_attn", "dt_seq", "interest_clock"}:
        dts_pad = torch.randint(1, num_dt_buckets, (batch_size, window_size), device=device)
        return (items_pad, dts_pad, mask, pos, neg)

    if model_type == "hour":
        hours_pad = torch.randint(0, 24, (batch_size, window_size), device=device)
        return (items_pad, hours_pad, mask, pos, neg)

    if model_type in {"lic_v1", "lic_v1_decay", "lic_v1_timebias"}:
        long_window = max(long_window, short_len)
        items_long = torch.randint(1, num_items, (batch_size, long_window), device=device)
        dts_long = torch.randint(1, num_dt_buckets, (batch_size, long_window), device=device)
        mask_long = torch.ones((batch_size, long_window), dtype=torch.bool, device=device)
        items_short = items_long[:, -short_len:]
        dts_short = dts_long[:, -short_len:]
        mask_short = torch.ones((batch_size, short_len), dtype=torch.bool, device=device)
        return (items_long, dts_long, mask_long, items_short, dts_short, mask_short, pos, neg)

    return (items_pad, mask, pos, neg)


def estimate_flops(model, forward_fn, inputs):
    try:
        with torch.profiler.profile(with_flops=True) as prof:
            forward_fn(*inputs)
        flops = 0
        for evt in prof.key_averages():
            if evt.flops is not None:
                flops += evt.flops
        return flops
    except Exception:
        return None


def measure_latency(forward_fn, inputs, warmup, repeats, device):
    for _ in range(warmup):
        forward_fn(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        forward_fn(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    return avg


def main():
    args = parse_args()
    device = torch.device(args.device)
    models = DEFAULT_MODELS if args.model == "all" else [args.model]

    results = {}
    for model_type in models:
        model = build_model(model_type, args.num_items, args.num_dt_buckets, args.emb_dim).to(device)
        model.eval()

        inputs = build_inputs(
            model_type,
            args.batch_size,
            args.window_size,
            args.long_window,
            args.short_len,
            args.num_items,
            args.num_dt_buckets,
            device
        )

        def forward_fn(*batch):
            with torch.no_grad():
                if model_type in {"mfseq", "dt_seq", "hour", "dt_attn", "interest_clock"}:
                    return model(*batch)
                return model(*batch)

        latency_s = measure_latency(forward_fn, inputs, args.warmup, args.repeats, device)
        throughput = args.batch_size / max(latency_s, 1e-9)
        params = sum(p.numel() for p in model.parameters())
        flops = estimate_flops(model, forward_fn, inputs)

        results[model_type] = {
            "latency_ms": latency_s * 1000.0,
            "throughput_sps": throughput,
            "params": params,
            "flops": flops
        }

    # 输出与保存
    for model_type, metrics in results.items():
        print(f"{model_type}: "
              f"latency={metrics['latency_ms']:.3f} ms, "
              f"throughput={metrics['throughput_sps']:.2f} samples/s, "
              f"params={metrics['params']}, "
              f"flops={metrics['flops']}")

    output_path = args.output
    if not output_path:
        out_dir = Path(__file__).resolve().parents[1] / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "benchmark.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
