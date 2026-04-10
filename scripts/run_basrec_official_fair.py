import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASREC_ROOT = ROOT / 'BASRec'


def _run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        raise RuntimeError(f'command failed: {cmd}\n{tail}')
    return proc.stdout


def _extract_last_metrics(text):
    best = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('{') and 'HIT@10' in s and 'NDCG@10' in s:
            try:
                d = ast.literal_eval(s)
                best = d
            except Exception:
                continue
    if best is None:
        raise RuntimeError('failed to parse metrics from BASRec output')
    out = {
        'hit@10': float(best.get('HIT@10')),
        'ndcg@10': float(best.get('NDCG@10')),
    }
    if 'MRR@10' in best:
        out['mrr@10'] = float(best.get('MRR@10'))
    elif 'MRR' in best:
        out['mrr@10'] = float(best.get('MRR'))
    return out


def _mean_std(values):
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        std = 0.0
    else:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = var ** 0.5
    return {'mean': mean, 'std': std, 'n': n}


def _print_summary(model, dataset, summary):
    print(f'[SUMMARY] {model} @ {dataset}')
    for k in ('hit@10', 'ndcg@10', 'mrr@10'):
        v = summary.get(k)
        if not isinstance(v, dict) or v.get('mean') is None:
            continue
        print(f"  {k}: mean={v['mean']:.6f}, std={v['std']:.6f}, n={v['n']}")


def _run_one_seed(args, seed):
    no_cuda = [] if args.device == 'gpu' else ['--no_cuda']
    gpu_id = str(args.gpu_id) if args.gpu_id is not None else '0'
    eval_seed = args.eval_seed if args.eval_seed is not None else seed

    cmd = [
        sys.executable, 'main.py',
        '--data_dir', 'data/',
        '--data_name', args.dataset,
        '--seed', str(seed),
        '--model_idx', str(seed),
        '--gpu_id', gpu_id,
        '--output_dir', 'output_fair/',
        '--epochs', str(args.epochs),
        '--patience', str(args.patience),
        '--batch_size', str(args.batch_size),
        '--max_seq_length', str(args.max_seq_length),
        '--start_valid', str(args.start_valid),
        '--pretrain_epoch', str(args.pretrain_epoch),
        '--eval_mode', args.eval_mode,
        '--num_neg', str(args.num_neg),
        '--eval_seed', str(eval_seed),
    ] + no_cuda

    out = _run(cmd, BASREC_ROOT)
    metrics = _extract_last_metrics(out)
    return {'seed': seed, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description='Fair BASRec official run with shared budget/protocol.')
    parser.add_argument('--dataset', required=True, help='Dataset alias, BASRec data name, or "all".')
    parser.add_argument('--seed', type=int, default=None, help='Run single seed. If omitted, uses --seeds.')
    parser.add_argument('--seeds', default='42,2023,2024')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_seq_length', type=int, default=32)
    parser.add_argument('--start_valid', type=int, default=0)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--eval_mode', default='uni100', choices=['uni100', 'full'])
    parser.add_argument('--num_neg', type=int, default=100)
    parser.add_argument('--eval_seed', type=int, default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    dataset_alias = {
        'beauty_ours': 'Beauty_ours',
        'amazon-beauty-5core-subset': 'Beauty_ours',
        'amazon-beauty-5core-subset-full': 'Beauty_ours',
        'beautylong_ours': 'BeautyLong_ours',
        'amazon-beauty-5core-longseq': 'BeautyLong_ours',
        'amazon-beauty-5core-longseq-full': 'BeautyLong_ours',
        'electronics_ours': 'Electronics_ours',
        'amazon-electronics-5core-subset': 'Electronics_ours',
        'amazon-electronics-5core-subset-full': 'Electronics_ours',
        'sports_ours': 'Sports_ours',
        'amazon-sports-5core-subset': 'Sports_ours',
        'amazon-sports-5core-subset-full': 'Sports_ours',
        'sportslong_ours': 'SportsLong_ours',
        'amazon-sports-5core-longseq': 'SportsLong_ours',
        'amazon-sports-5core-longseq-full': 'SportsLong_ours',
        'toys_ours': 'Toys_ours',
        'amazon-toys-5core-subset': 'Toys_ours',
        'amazon-toys-5core-subset-full': 'Toys_ours',
        'toyslong_ours': 'ToysLong_ours',
        'amazon-toys-5core-longseq': 'ToysLong_ours',
        'amazon-toys-5core-longseq-full': 'ToysLong_ours',
        'toys50k_ours': 'Toys50K_ours',
        'amazon-toys-5core-subset-50k': 'Toys50K_ours',
        'amazon-toys-5core-subset-50k-full': 'Toys50K_ours',
        'ml100k_ours': 'ML100K_ours',
        'ml-100k': 'ML100K_ours',
        'ml-100k-full': 'ML100K_ours',
    }
    key = args.dataset.strip().lower()
    if key == 'all':
        datasets = ['ML100K_ours', 'Beauty_ours', 'BeautyLong_ours', 'Electronics_ours', 'Sports_ours', 'SportsLong_ours', 'Toys_ours', 'ToysLong_ours', 'Toys50K_ours']
    else:
        datasets = [dataset_alias.get(key, args.dataset)]

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        if not seeds:
            raise ValueError('No valid seeds provided.')

    all_results = []
    for dataset in datasets:
        args.dataset = dataset
        runs = []
        for seed in seeds:
            print(f'[RUN] basrec_official @ {args.dataset} seed={seed}')
            runs.append(_run_one_seed(args, seed))

        hit = _mean_std([r['metrics']['hit@10'] for r in runs])
        ndcg = _mean_std([r['metrics']['ndcg@10'] for r in runs])
        mrr = _mean_std([r['metrics']['mrr@10'] for r in runs if 'mrr@10' in r['metrics']])
        all_results.append({
            'dataset': args.dataset,
            'epochs': args.epochs,
            'patience': args.patience,
            'max_seq_length': args.max_seq_length,
            'eval_mode': args.eval_mode,
            'runs': runs,
            'summary': {
                'hit@10': hit,
                'ndcg@10': ndcg,
                'mrr@10': mrr,
            },
        })
        _print_summary('basrec_official', args.dataset, all_results[-1]['summary'])

    if len(all_results) == 1:
        result = all_results[0]
    else:
        result = {
            'datasets': datasets,
            'results': all_results,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
