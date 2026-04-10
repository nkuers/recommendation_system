import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from recbole.config import Config
from recbole.data import create_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.timeweaver import TimeWeaver



DATASET_CONFIGS = {
    "ml-100k": "recbole_config/ml-100k.yaml",
    "amazon-beauty-5core-subset": "recbole_config/amazon-beauty-5core-subset.yaml",
    "amazon-electronics-5core-subset": "recbole_config/amazon-electronics-5core-subset.yaml",
    "amazon-sports-5core-longseq": "recbole_config/amazon-sports-5core-longseq.yaml",
    "amazon-toys-5core-longseq": "recbole_config/amazon-toys-5core-longseq.yaml",
}

MODEL_CONFIG = "recbole_config/timeweaver.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TimeWeaver efficiency.")
    parser.add_argument("--model", default="timeweaver", choices=["timeweaver"])
    parser.add_argument("--dataset", required=True, choices=DATASET_CONFIGS.keys())
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--mode", default="full_sort", choices=["forward", "predict", "full_sort"])
    parser.add_argument("--timeweaver-variant", default="base",
                        choices=["base", "cont", "gate_discrete", "gate", "aug_only", "aug"])
    parser.add_argument("--output", default="", help="Output json path.")
    return parser.parse_args()


def variant_config(variant):
    base = {
        "use_continuous_time": False,
        "use_time_gate": False,
        "use_time_augmentation": False,
    }
    if variant == "cont":
        base.update({"use_continuous_time": True})
    elif variant == "gate_discrete":
        base.update({"use_time_gate": True})
    elif variant == "gate":
        base.update({"use_continuous_time": True, "use_time_gate": True})
    elif variant == "aug_only":
        base.update({"use_time_augmentation": True})
    elif variant == "aug":
        base.update({"use_continuous_time": True, "use_time_gate": True, "use_time_augmentation": True})
    return base


def build_model(dataset_name, device, model_name, variant):
    config_files = [DATASET_CONFIGS[dataset_name], MODEL_CONFIG]
    config_dict = {
        "device": device,
        "train_neg_sample_args": None,
        "neg_sampling": None,
        **variant_config(variant),
    }
    config = Config(
        model="TimeWeaver",
        dataset=dataset_name,
        config_file_list=config_files,
        config_dict=config_dict,
    )
    dataset = create_dataset(config)
    model = TimeWeaver(config, dataset).to(device)
    model.eval()
    return model, config, dataset


def build_inputs(model, config, batch_size, device):
    seq_len = config["MAX_ITEM_LIST_LENGTH"]
    n_items = model.n_items

    item_seq = torch.randint(1, n_items, (batch_size, seq_len), device=device)
    intervals = torch.randint(0, 3600, (batch_size, seq_len), device=device)
    time_seq = torch.cumsum(intervals, dim=1)
    item_seq_len = torch.randint(1, seq_len + 1, (batch_size,), device=device)
    test_item = torch.randint(1, n_items, (batch_size,), device=device)
    pos_item = torch.randint(1, n_items, (batch_size,), device=device)
    neg_item = torch.randint(1, n_items, (batch_size,), device=device)

    interaction = {
        model.ITEM_SEQ: item_seq,
        model.TIME_SEQ: time_seq,
        model.ITEM_SEQ_LEN: item_seq_len,
        model.ITEM_ID: test_item,
        model.POS_ITEM_ID: pos_item,
        model.NEG_ITEM_ID: neg_item,
    }
    return interaction


def estimate_flops(forward_fn):
    try:
        with torch.profiler.profile(with_flops=True) as prof:
            forward_fn()
        flops = 0
        for evt in prof.key_averages():
            if evt.flops is not None:
                flops += evt.flops
        return flops
    except Exception:
        return None


def measure_latency(forward_fn, warmup, repeats, device):
    for _ in range(warmup):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, config, _ = build_model(args.dataset, device, args.model, args.timeweaver_variant)
    interaction = build_inputs(model, config, args.batch_size, device)

    def forward_fn():
        with torch.no_grad():
            if args.mode == "forward":
                return model.forward(interaction[model.ITEM_SEQ], interaction[model.TIME_SEQ])
            if args.mode == "predict":
                return model.predict(interaction)
            return model.full_sort_predict(interaction)

    latency_s = measure_latency(forward_fn, args.warmup, args.repeats, device)
    throughput = args.batch_size / max(latency_s, 1e-9)
    params = sum(p.numel() for p in model.parameters())
    flops = estimate_flops(forward_fn)

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "variant": args.timeweaver_variant,
        "mode": args.mode,
        "latency_ms": latency_s * 1000.0,
        "throughput_sps": throughput,
        "params": params,
        "flops": flops,
    }

    output_path = args.output
    if not output_path:
        out_dir = ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "benchmark_timeweaver.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2))
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
