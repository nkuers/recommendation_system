# scripts/run_grid.py
import argparse
import json
import subprocess
import sys
import types
import tempfile
from pathlib import Path

import torch
import pandas as pd
from recbole.quick_start import run_recbole

# Optional exlib deps are imported by RecBole on model discovery; provide safe stubs when absent.
for _mod in ("xgboost",):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

if "xgboost" in sys.modules:
    _xgb = sys.modules["xgboost"]
    if not hasattr(_xgb, "Booster"):
        class _DummyBooster:
            pass
        class _DummyDMatrix:
            def __init__(self, *args, **kwargs):
                pass
        def _dummy_train(*args, **kwargs):
            raise RuntimeError("xgboost is not installed; exlib models are unavailable")
        _xgb.Booster = _DummyBooster
        _xgb.DMatrix = _DummyDMatrix
        _xgb.train = _dummy_train


_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

# Pandas/Dask compatibility shim for some LightGBM transitively imported by RecBole.
if not hasattr(pd.core.strings, "StringMethods"):
    try:
        from pandas.core.strings.accessor import StringMethods as _PandasStringMethods
        pd.core.strings.StringMethods = _PandasStringMethods
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIGS = {
    "basrec": "recbole_config/basrec.yaml",
    "caser": "recbole_config/caser.yaml",
    "gru4rec": "recbole_config/gru4rec.yaml",
    "lru": "recbole_config/lru.yaml",
    "mlm4rec": "recbole_config/mlm4rec.yaml",
    "mlm4rec_official": "recbole_config/mlm4rec.yaml",
    "timeweaver": "recbole_config/timeweaver.yaml",
    "tisasrec": "recbole_config/tisasrec.yaml",
    "lrurec": "recbole_config/lrurec.yaml",
    "lrurec_official": "recbole_config/lru.yaml",
    "idurl": "recbole_config/sasrec_idurl.yaml",
    "idea": "recbole_config/idea_gru.yaml",
}

DATASET_CONFIGS = {
    "ml-100k": "recbole_config/ml-100k.yaml",
    "ml-100k-full": "recbole_config/ml-100k-full.yaml",
    "amazon-beauty-5core-subset": "recbole_config/amazon-beauty-5core-subset.yaml",
    "amazon-beauty-5core-subset-full": "recbole_config/amazon-beauty-5core-subset-full.yaml",
    "amazon-beauty-5core-longseq": "recbole_config/amazon-beauty-5core-longseq.yaml",
    "amazon-beauty-5core-longseq-full": "recbole_config/amazon-beauty-5core-longseq-full.yaml",
    "amazon-electronics-5core-subset": "recbole_config/amazon-electronics-5core-subset.yaml",
    "amazon-electronics-5core-subset-full": "recbole_config/amazon-electronics-5core-subset-full.yaml",
    "amazon-sports-5core-subset": "recbole_config/amazon-sports-5core-subset.yaml",
    "amazon-sports-5core-subset-full": "recbole_config/amazon-sports-5core-subset-full.yaml",
    "amazon-sports-5core-longseq": "recbole_config/amazon-sports-5core-longseq.yaml",
    "amazon-sports-5core-longseq-full": "recbole_config/amazon-sports-5core-longseq-full.yaml",
    "amazon-toys-5core-subset": "recbole_config/amazon-toys-5core-subset.yaml",
    "amazon-toys-5core-subset-full": "recbole_config/amazon-toys-5core-subset-full.yaml",
    "amazon-toys-5core-longseq": "recbole_config/amazon-toys-5core-longseq.yaml",
    "amazon-toys-5core-longseq-full": "recbole_config/amazon-toys-5core-longseq-full.yaml",
    "amazon-toys-5core-subset-50k": "recbole_config/amazon-toys-5core-subset-50k.yaml",
    "amazon-toys-5core-subset-50k-full": "recbole_config/amazon-toys-5core-subset-50k-full.yaml",
}

DATASET_RUNTIME_NAMES = {
    "ml-100k-full": "ml-100k",
    "amazon-beauty-5core-subset-full": "amazon-beauty-5core-subset",
    "amazon-beauty-5core-longseq-full": "amazon-beauty-5core-longseq",
    "amazon-electronics-5core-subset-full": "amazon-electronics-5core-subset",
    "amazon-sports-5core-subset-full": "amazon-sports-5core-subset",
    "amazon-sports-5core-longseq-full": "amazon-sports-5core-longseq",
    "amazon-toys-5core-subset-full": "amazon-toys-5core-subset",
    "amazon-toys-5core-longseq-full": "amazon-toys-5core-longseq",
    "amazon-toys-5core-subset-50k-full": "amazon-toys-5core-subset-50k",
}

IDURL_DATASET_CONFIGS = {
    "ml-100k": "recbole_config/ml-100k-idurl.yaml",
    "amazon-beauty-5core-subset": "recbole_config/amazon-beauty-5core-subset-idurl.yaml",
    "amazon-electronics-5core-subset": "recbole_config/amazon-electronics-5core-subset-idurl.yaml",
}

IDURL_DATASET_NAMES = {
    "ml-100k": "ml-100k-idurl",
    "amazon-beauty-5core-subset": "amazon-beauty-5core-subset-idurl",
    "amazon-electronics-5core-subset": "amazon-electronics-5core-subset-idurl",
}

# Model-specific dataset compatibility (can override with --force)
MODEL_DATASET_ALLOW = {
    "basrec": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "caser": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "gru4rec": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "lru": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "mlm4rec": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "mlm4rec_official": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "timeweaver": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "tisasrec": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "lrurec": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "lrurec_official": {"ml-100k", "ml-100k-full", "amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full", "amazon-beauty-5core-longseq", "amazon-beauty-5core-longseq-full", "amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full", "amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full", "amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full", "amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"},
    "idurl": {"ml-100k", "amazon-beauty-5core-subset", "amazon-electronics-5core-subset"},
    "idea": {"ml-100k", "amazon-beauty-5core-subset", "amazon-electronics-5core-subset"},
}


def parse_list(value):
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _collect_metrics(result_block):
    if not isinstance(result_block, dict):
        return None
    test_result = result_block.get("test_result")
    if isinstance(test_result, dict):
        return test_result
    best_valid_result = result_block.get("best_valid_result")
    if isinstance(best_valid_result, dict):
        return best_valid_result
    return None


def summarize_results(results, out_path=None):
    by_group = {}
    for entry in results.values():
        if not isinstance(entry, dict):
            continue
        if "error" in entry:
            continue
        model = entry.get("model")
        dataset = entry.get("dataset")
        variant = entry.get("variant")
        tw_overrides = entry.get("tw_overrides") or {}
        tw_tag = tw_overrides.get("tag")
        result = entry.get("result")
        metrics = _collect_metrics(result)
        if not model or not dataset or not metrics:
            continue
        if model == "timeweaver" and variant:
            group_key = (model, dataset, variant, tw_tag or "")
        else:
            group_key = (model, dataset)
        by_group.setdefault(group_key, []).append(metrics)

    if not by_group:
        return

    summary_out = {}
    for group_key, metrics_list in by_group.items():
        if len(group_key) == 4:
            model, dataset, variant, tw_tag = group_key
        elif len(group_key) == 3:
            model, dataset, variant = group_key
            tw_tag = ""
        else:
            model, dataset = group_key
            variant = None
            tw_tag = ""
        metric_keys = [k for k in metrics_list[0].keys() if isinstance(metrics_list[0][k], (int, float))]
        agg = {}
        for k in metric_keys:
            vals = [m.get(k) for m in metrics_list if isinstance(m.get(k), (int, float))]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
            agg[k] = {"mean": mean, "std": std, "n": len(vals)}

        if not agg:
            continue
        if variant:
            summary_key = f"{model}::{dataset}::{variant}"
            if tw_tag:
                summary_key = f"{summary_key}::{tw_tag}"
                print(f"[SUMMARY] {model}({variant}/{tw_tag}) @ {dataset}")
            else:
                print(f"[SUMMARY] {model}({variant}) @ {dataset}")
            summary_out[summary_key] = agg
        else:
            summary_out[f"{model}::{dataset}"] = agg
            print(f"[SUMMARY] {model} @ {dataset}")
        for k, v in agg.items():
            print(f"  {k}: mean={v['mean']:.6f}, std={v['std']:.6f}, n={v['n']}")

    if out_path is not None:
        save_json(out_path, summary_out)



def _ensure_model_registered(model):
    register_map = {
        "basrec": ("models/basrec.py", "BASRec"),
        "timeweaver": ("models/timeweaver.py", "TimeWeaver"),
        "tisasrec": ("models/tisasrec.py", "TiSASRec"),
        "mlm4rec": ("models/mlm4rec.py", "MLM4Rec"),
        "lru": ("models/LRU.py", "LRURec"),
        "lrurec": ("models/lrurec.py", "LRURec"),
        "idurl": ("models/sasrec_idurl.py", "SASRec_IDURL"),
        "idea": ("models/ideagru.py", "IDEAGRU"),
    }
    if model not in register_map:
        return
    model_file, model_name = register_map[model]
    cmd = [
        sys.executable,
        str(ROOT / "scripts/register_recbole_model.py"),
        "--model_file",
        str(ROOT / model_file),
        "--model_name",
        model_name,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"register model failed for {model}: {proc.stderr[-1200:] or proc.stdout[-1200:]}"
        )



def _run_idea_two_stage(dataset, config_list, seed, device, epochs, paper_mode=False):
    tmp_dir = ROOT / 'results' / '_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='idea_two_stage_', suffix='.json', dir=str(tmp_dir), delete=False) as tf:
        out_file = Path(tf.name)

    cmd = [
        sys.executable,
        str(ROOT / 'scripts/run_idea_two_stage.py'),
        '--dataset',
        dataset,
        '--config_files',
        ','.join(config_list),
        '--seed',
        str(seed),
        '--device',
        device,
        '--epochs',
        str(epochs),
        '--output',
        str(out_file),
    ]
    if paper_mode:
        cmd.append('--paper-mode')

    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1500:] or proc.stdout[-1500:] or f'idea two-stage failed code={proc.returncode}')

    if not out_file.exists():
        raise RuntimeError('idea two-stage did not produce output json')

    try:
        data = json.loads(out_file.read_text(encoding='utf-8'))
    finally:
        try:
            out_file.unlink(missing_ok=True)
        except Exception:
            pass

    if not isinstance(data, dict) or 'test_result' not in data:
        raise RuntimeError('idea two-stage output format invalid')
    return data


def _run_lrurec_official(dataset, seed, device, epochs):
    tmp_dir = ROOT / 'results' / '_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='lrurec_official_', suffix='.json', dir=str(tmp_dir), delete=False) as tf:
        out_file = Path(tf.name)

    cmd = [
        sys.executable,
        str(ROOT / 'scripts/run_lrurec_official_fair.py'),
        '--dataset',
        dataset,
        '--seed',
        str(seed),
        '--device',
        device,
        '--epochs',
        str(epochs),
        '--patience',
        '10',
        '--batch_size',
        '256',
        '--eval_batch_size',
        '512',
        '--max_seq_length',
        '32',
        '--hidden_size',
        '64',
        '--num_layers',
        '2',
        '--dropout',
        '0.2',
        '--attn_dropout',
        '0.2',
        '--learning_rate',
        '0.001',
        '--weight_decay',
        '0.01',
        '--output',
        str(out_file),
    ]

    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1500:] or proc.stdout[-1500:] or f'lrurec official failed code={proc.returncode}')

    if not out_file.exists():
        raise RuntimeError('lrurec official did not produce output json')

    try:
        data = json.loads(out_file.read_text(encoding='utf-8'))
    finally:
        try:
            out_file.unlink(missing_ok=True)
        except Exception:
            pass

    runs = data.get('runs') or []
    if not runs or not isinstance(runs[0], dict):
        raise RuntimeError('lrurec official output format invalid: missing runs')
    metrics = runs[0].get('metrics') or {}
    if not metrics:
        raise RuntimeError('lrurec official output format invalid: missing metrics')

    return {
        'test_result': {
            'hit@10': float(metrics.get('hit@10')),
            'ndcg@10': float(metrics.get('ndcg@10')),
            'mrr@10': float(metrics.get('mrr@10')),
        }
    }


def _run_mlm4rec_official(dataset, seed, device, epochs):
    tmp_dir = ROOT / 'results' / '_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='mlm4rec_official_', suffix='.json', dir=str(tmp_dir), delete=False) as tf:
        out_file = Path(tf.name)

    cmd = [
        sys.executable,
        str(ROOT / 'scripts/run_mlm4rec_official_fair.py'),
        '--dataset',
        dataset,
        '--seed',
        str(seed),
        '--device',
        device,
        '--epochs',
        str(epochs),
        '--patience',
        '10',
        '--batch_size',
        '256',
        '--max_seq_length',
        '32',
        '--hidden_size',
        '64',
        '--num_layers',
        '2',
        '--hidden_dropout',
        '0.5',
        '--attn_dropout',
        '0.5',
        '--learning_rate',
        '0.001',
        '--weight_decay',
        '0.0',
        '--eval_mode',
        'uni100',
        '--num_neg',
        '100',
        '--output',
        str(out_file),
    ]

    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1500:] or proc.stdout[-1500:] or f'mlm4rec official failed code={proc.returncode}')

    if not out_file.exists():
        raise RuntimeError('mlm4rec official did not produce output json')

    try:
        data = json.loads(out_file.read_text(encoding='utf-8'))
    finally:
        try:
            out_file.unlink(missing_ok=True)
        except Exception:
            pass

    runs = data.get('runs') or []
    if not runs or not isinstance(runs[0], dict):
        raise RuntimeError('mlm4rec official output format invalid: missing runs')
    metrics = runs[0].get('metrics') or {}
    if not metrics:
        raise RuntimeError('mlm4rec official output format invalid: missing metrics')

    return {
        'test_result': {
            'hit@10': float(metrics.get('hit@10')),
            'ndcg@10': float(metrics.get('ndcg@10')),
            'mrr@10': float(metrics.get('mrr@10')),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run RecBole models on datasets with repeats.")
    parser.add_argument("--models", default="all", help="Comma list or 'all'")
    parser.add_argument("--datasets", default="all", help="Comma list or 'all'")
    parser.add_argument("--seeds", default="42,2023,2024", help="Comma list of seeds")
    parser.add_argument("--device", default="gpu", help="gpu or cpu")
    parser.add_argument("--timeweaver-variant", default="base",
                        choices=["base", "cont", "gate_discrete", "gate", "aug_only", "aug"],
                        help="TimeWeaver ablation switch.")
    parser.add_argument("--tw-gate-hidden", type=int, default=None, help="Override time_gate_hidden for TimeWeaver.")
    parser.add_argument("--tw-aug-strength", type=float, default=None, help="Override time_aug_strength for TimeWeaver.")
    parser.add_argument("--tw-aug-prob", type=float, default=None, help="Override time_aug_prob for TimeWeaver.")
    parser.add_argument("--tw-time-aug-mode", type=str, default=None, choices=["mix", "median", "jitter"],
                        help="Override time_aug_mode for TimeWeaver.")
    parser.add_argument("--tw-time-aug-warmup", type=int, default=None, help="Override time_aug_warmup_steps for TimeWeaver.")
    parser.add_argument("--tw-time-aug-ramp", type=int, default=None, help="Override time_aug_ramp_steps for TimeWeaver.")
    parser.add_argument("--tw-contrastive-weight", type=float, default=None, help="Override contrastive_weight for TimeWeaver.")
    parser.add_argument("--tw-contrastive-temp", type=float, default=None, help="Override contrastive_temp for TimeWeaver.")
    parser.add_argument("--tw-ema-decay", type=float, default=None, help="Override ema_decay for TimeWeaver.")
    parser.add_argument("--tw-tcn-kernel", type=int, default=None, help="Override tcn_kernel_size for TimeWeaver.")
    parser.add_argument("--tw-ffn-hidden", type=int, default=None, help="Override ffn_hidden_size for TimeWeaver.")
    parser.add_argument("--tw-hidden-dropout", type=float, default=None, help="Override hidden_dropout_prob for TimeWeaver.")
    parser.add_argument("--tw-num-layers", type=int, default=None, help="Override num_layers for TimeWeaver.")
    parser.add_argument("--tw-time-bucket", type=int, default=None, help="Override time_bucket_size for TimeWeaver.")
    parser.add_argument("--tag", default="", help="Optional run tag (added to result key).")
    parser.add_argument("--force", action="store_true", help="Force unsupported model-dataset pairs")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--skip-if-done", action="store_true", help="Skip if result exists in output")
    parser.add_argument("--output", default="results/run_grid.json", help="Output json path")
    parser.add_argument("--epochs-override", type=int, default=None, help="Override epochs for all models.")
    parser.add_argument("--idea-paper-mode", action="store_true",
                        help="Use IDEA fixed paper schedule (default is fair early-stop mode).")
    parser.add_argument("--no-summary", action="store_true", help="Skip writing summary json")
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.models == "all" else parse_list(args.models)
    datasets = list(DATASET_CONFIGS.keys()) if args.datasets == "all" else parse_list(args.datasets)
    seeds = [int(s) for s in parse_list(args.seeds)]

    out_path = ROOT / args.output
    results = load_json(out_path)

    plan = []
    for model in models:
        if model not in MODEL_CONFIGS:
            raise SystemExit(f"Unknown model: {model}")
        for dataset in datasets:
            if dataset not in DATASET_CONFIGS:
                raise SystemExit(f"Unknown dataset: {dataset}")
            allowed = MODEL_DATASET_ALLOW.get(model, set())
            if allowed and dataset not in allowed and not args.force:
                print(f"[SKIP] {model} not configured for {dataset} (use --force to run)")
                continue
            for seed in seeds:
                plan.append((model, dataset, seed))

    if args.dry_run:
        print("[PLAN]")
        for model, dataset, seed in plan:
            print(f"  {model} @ {dataset} seed={seed}")
        return

    for model, dataset, seed in plan:
        key = f"{model}::{dataset}::{seed}"
        if model == "timeweaver":
            key = f"{model}::{dataset}::{args.timeweaver_variant}::{seed}"
        if args.tag:
            key = f"{key}::{args.tag}"
        if args.skip_if_done and key in results:
            print(f"[SKIP] done: {key}")
            continue

        _ensure_model_registered(model)
        dataset_cfg = DATASET_CONFIGS[dataset]
        run_dataset = DATASET_RUNTIME_NAMES.get(dataset, dataset)
        if model == "idurl":
            dataset_cfg = IDURL_DATASET_CONFIGS.get(dataset, dataset_cfg)
            run_dataset = IDURL_DATASET_NAMES.get(dataset, dataset)
        config_list = [dataset_cfg, MODEL_CONFIGS[model]]
        config_dict = {"seed": seed, "device": args.device}
        if args.epochs_override is not None:
            config_dict["epochs"] = args.epochs_override

        # CE loss in RecBole should not use negative sampling
        if model in {"basrec", "timeweaver", "lrurec", "gru4rec", "caser", "lru", "idurl", "idea"}:
            config_dict["train_neg_sample_args"] = None
            config_dict["neg_sampling"] = None
        if model == "tisasrec":
            if dataset in {"ml-100k", "ml-100k-full"}:
                config_dict["time_span"] = 2048
            elif dataset in {"amazon-beauty-5core-subset", "amazon-beauty-5core-subset-full"}:
                config_dict["time_span"] = 512
            elif dataset in {"amazon-electronics-5core-subset", "amazon-electronics-5core-subset-full"}:
                config_dict["time_span"] = 512
            elif dataset in {"amazon-sports-5core-subset", "amazon-sports-5core-subset-full", "amazon-sports-5core-longseq", "amazon-sports-5core-longseq-full"}:
                config_dict["time_span"] = 512
            elif dataset in {"amazon-toys-5core-subset", "amazon-toys-5core-subset-full", "amazon-toys-5core-longseq", "amazon-toys-5core-longseq-full"}:
                config_dict["time_span"] = 512
            elif dataset in {"amazon-toys-5core-subset-50k", "amazon-toys-5core-subset-50k-full"}:
                config_dict["time_span"] = 512
        if model == "timeweaver":
            if args.timeweaver_variant == "base":
                config_dict.update({
                    "use_continuous_time": False,
                    "use_time_gate": False,
                    "use_time_augmentation": False,
                })
            elif args.timeweaver_variant == "cont":
                config_dict.update({
                    "use_continuous_time": True,
                    "use_time_gate": False,
                    "use_time_augmentation": False,
                })
            elif args.timeweaver_variant == "gate_discrete":
                config_dict.update({
                    "use_continuous_time": False,
                    "use_time_gate": True,
                    "use_time_augmentation": False,
                })
            elif args.timeweaver_variant == "gate":
                config_dict.update({
                    "use_continuous_time": True,
                    "use_time_gate": True,
                    "use_time_augmentation": False,
                })
            elif args.timeweaver_variant == "aug_only":
                config_dict.update({
                    "use_continuous_time": False,
                    "use_time_gate": False,
                    "use_time_augmentation": True,
                })
            elif args.timeweaver_variant == "aug":
                config_dict.update({
                    "use_continuous_time": True,
                    "use_time_gate": True,
                    "use_time_augmentation": True,
                })
            if args.tw_gate_hidden is not None:
                config_dict["time_gate_hidden"] = args.tw_gate_hidden
            if args.tw_aug_strength is not None:
                config_dict["time_aug_strength"] = args.tw_aug_strength
            if args.tw_aug_prob is not None:
                config_dict["time_aug_prob"] = args.tw_aug_prob
            if args.tw_time_aug_mode is not None:
                config_dict["time_aug_mode"] = args.tw_time_aug_mode
            if args.tw_time_aug_warmup is not None:
                config_dict["time_aug_warmup_steps"] = args.tw_time_aug_warmup
            if args.tw_time_aug_ramp is not None:
                config_dict["time_aug_ramp_steps"] = args.tw_time_aug_ramp
            if args.tw_contrastive_weight is not None:
                config_dict["contrastive_weight"] = args.tw_contrastive_weight
            if args.tw_contrastive_temp is not None:
                config_dict["contrastive_temp"] = args.tw_contrastive_temp
            if args.tw_ema_decay is not None:
                config_dict["ema_decay"] = args.tw_ema_decay
            if args.tw_tcn_kernel is not None:
                config_dict["tcn_kernel_size"] = args.tw_tcn_kernel
            if args.tw_ffn_hidden is not None:
                config_dict["ffn_hidden_size"] = args.tw_ffn_hidden
            if args.tw_hidden_dropout is not None:
                config_dict["hidden_dropout_prob"] = args.tw_hidden_dropout
            if args.tw_num_layers is not None:
                config_dict["num_layers"] = args.tw_num_layers
            if args.tw_time_bucket is not None:
                config_dict["time_bucket_size"] = args.tw_time_bucket

        if model == "timeweaver":
            print(f"[RUN] {model}({args.timeweaver_variant}) @ {dataset} seed={seed}")
        else:
            print(f"[RUN] {model} @ {dataset} seed={seed}")
        model_name = {
            "basrec": "BASRec",
            "caser": "Caser",
            "gru4rec": "GRU4Rec",
            "lru": "LRURec",
            "mlm4rec": "MLM4Rec",
            "mlm4rec_official": "MLM4Rec",
            "timeweaver": "TimeWeaver",
            "tisasrec": "TiSASRec",
            "lrurec": "LRURec",
            "lrurec_official": "LRURec",
            "idurl": "SASRec_IDURL",
            "idea": "IDEAGRU",
        }[model]
        try:
            if model == "idea":
                result = _run_idea_two_stage(
                    dataset=dataset,
                    config_list=config_list,
                    seed=seed,
                    device=args.device,
                    epochs=config_dict.get("epochs", 100),
                    paper_mode=args.idea_paper_mode,
                )
            elif model == "lrurec_official":
                result = _run_lrurec_official(
                    dataset=dataset,
                    seed=seed,
                    device=args.device,
                    epochs=config_dict.get("epochs", 100),
                )
            elif model == "mlm4rec_official":
                result = _run_mlm4rec_official(
                    dataset=dataset,
                    seed=seed,
                    device=args.device,
                    epochs=config_dict.get("epochs", 100),
                )
            else:
                result = run_recbole(
                    model=model_name,
                    dataset=run_dataset,
                    config_file_list=config_list,
                    config_dict=config_dict,
                )
            entry = {
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "result": result,
            }
            if model == "timeweaver":
                entry["variant"] = args.timeweaver_variant
                entry["tw_overrides"] = {
                    "time_gate_hidden": args.tw_gate_hidden,
                    "time_aug_strength": args.tw_aug_strength,
                    "time_aug_prob": args.tw_aug_prob,
                    "time_aug_mode": args.tw_time_aug_mode,
                    "time_aug_warmup_steps": args.tw_time_aug_warmup,
                    "time_aug_ramp_steps": args.tw_time_aug_ramp,
                    "contrastive_weight": args.tw_contrastive_weight,
                    "contrastive_temp": args.tw_contrastive_temp,
                    "tag": args.tag,
                }
            results[key] = entry
            save_json(out_path, results)
        except Exception as exc:
            entry = {
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "error": str(exc),
            }
            if model == "timeweaver":
                entry["variant"] = args.timeweaver_variant
                entry["tw_overrides"] = {
                    "time_gate_hidden": args.tw_gate_hidden,
                    "time_aug_strength": args.tw_aug_strength,
                    "time_aug_prob": args.tw_aug_prob,
                    "time_aug_mode": args.tw_time_aug_mode,
                    "time_aug_warmup_steps": args.tw_time_aug_warmup,
                    "time_aug_ramp_steps": args.tw_time_aug_ramp,
                    "contrastive_weight": args.tw_contrastive_weight,
                    "contrastive_temp": args.tw_contrastive_temp,
                    "tag": args.tag,
                }
            results[key] = entry
            save_json(out_path, results)
            print(f"[ERROR] {key}: {exc}")

    print(f"[INFO] saved results to {out_path}")
    if not args.no_summary:
        summary_path = out_path.with_name(out_path.stem + "_summary.json")
        summarize_results(results, summary_path)
        print(f"[INFO] summary saved to {summary_path}")


if __name__ == "__main__":
    main()


