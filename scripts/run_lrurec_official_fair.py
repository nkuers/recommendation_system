import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LRUREC_ROOT = ROOT / 'LRURec-master'


def _run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        raise RuntimeError(f'command failed: {cmd}\n{tail}')
    return proc.stdout


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


def _export_root(dataset, weight_decay, dropout, attn_dropout):
    name = f'{dataset}_{weight_decay}_{dropout}_{attn_dropout}'
    return LRUREC_ROOT / 'experiments' / 'lru' / name


def _extract_metrics(dataset, weight_decay, dropout, attn_dropout):
    path = _export_root(dataset, weight_decay, dropout, attn_dropout) / 'test_metrics.json'
    if not path.exists():
        raise RuntimeError(f'no test metrics found: {path}')
    d = json.loads(path.read_text(encoding='utf-8'))
    return {
        'hit@10': float(d['Recall@10']),
        'ndcg@10': float(d['NDCG@10']),
        'mrr@10': float(d['MRR@10']),
    }


def _run_one_seed(args, seed):
    cmd = [
        sys.executable,
        'train.py',
        '--dataset_code', args.dataset,
        '--seed', str(seed),
        '--num_epochs', str(args.epochs),
        '--early_stopping_patience', str(args.patience),
        '--val_strategy', 'epoch',
        '--train_batch_size', str(args.batch_size),
        '--val_batch_size', str(args.eval_batch_size),
        '--test_batch_size', str(args.eval_batch_size),
        '--bert_max_len', str(args.max_seq_length),
        '--bert_hidden_units', str(args.hidden_size),
        '--bert_num_blocks', str(args.num_layers),
        '--bert_dropout', str(args.dropout),
        '--bert_attn_dropout', str(args.attn_dropout),
        '--lr', str(args.learning_rate),
        '--weight_decay', str(args.weight_decay),
        '--num_workers', '0',
        '--device', 'cuda' if args.device == 'gpu' else 'cpu',
        '--eval_mode', args.eval_mode,
        '--num_neg', str(args.num_neg),
        '--eval_seed', str(args.eval_seed if args.eval_seed is not None else seed),
    ]
    _run(cmd, LRUREC_ROOT)
    metrics = _extract_metrics(args.dataset, args.weight_decay, args.dropout, args.attn_dropout)
    return {'seed': seed, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description='Fair official LRURec run with shared budget/protocol.')
    parser.add_argument('--dataset', required=True, help='Dataset alias, official code, or "all".')
    parser.add_argument('--seed', type=int, default=None, help='Run single seed. If omitted, uses --seeds.')
    parser.add_argument('--seeds', default='42,2023,2024')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--max_seq_length', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--eval_mode', default='uni100', choices=['uni100', 'full'])
    parser.add_argument('--num_neg', type=int, default=100)
    parser.add_argument('--eval_seed', type=int, default=None)
    parser.add_argument('--refresh_preprocessed', action='store_true', default=True)
    parser.add_argument('--no_refresh_preprocessed', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    dataset_alias = {
        'amazon-beauty-5core-subset': 'beauty_ours',
        'beauty_ours': 'beauty_ours',
        'amazon-beauty-5core-subset-full': 'beauty_ours',
        'amazon-beauty-5core-longseq': 'beautylong_ours',
        'beautylong_ours': 'beautylong_ours',
        'amazon-beauty-5core-longseq-full': 'beautylong_ours',
        'amazon-electronics-5core-subset': 'electronics_ours',
        'electronics_ours': 'electronics_ours',
        'amazon-electronics-5core-subset-full': 'electronics_ours',
        'amazon-sports-5core-subset': 'sports_ours',
        'sports_ours': 'sports_ours',
        'amazon-sports-5core-subset-full': 'sports_ours',
        'amazon-sports-5core-longseq': 'sportslong_ours',
        'sportslong_ours': 'sportslong_ours',
        'amazon-sports-5core-longseq-full': 'sportslong_ours',
        'amazon-toys-5core-subset': 'toys_ours',
        'toys_ours': 'toys_ours',
        'amazon-toys-5core-subset-full': 'toys_ours',
        'amazon-toys-5core-longseq': 'toyslong_ours',
        'toyslong_ours': 'toyslong_ours',
        'amazon-toys-5core-longseq-full': 'toyslong_ours',
        'amazon-toys-5core-subset-50k': 'toys50k_ours',
        'toys50k_ours': 'toys50k_ours',
        'amazon-toys-5core-subset-50k-full': 'toys50k_ours',
        'ml-100k': 'ml100k_ours',
        'ml100k_ours': 'ml100k_ours',
        'ml-100k-full': 'ml100k_ours',
    }
    key = args.dataset.strip().lower()
    if key == 'all':
        datasets = ['ml100k_ours', 'beauty_ours', 'beautylong_ours', 'electronics_ours', 'sports_ours', 'sportslong_ours', 'toys_ours', 'toyslong_ours', 'toys50k_ours']
    else:
        datasets = [dataset_alias.get(key, args.dataset)]
    if args.no_refresh_preprocessed:
        args.refresh_preprocessed = False

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        if not seeds:
            raise ValueError('No valid seeds provided.')

    all_results = []
    for dataset in datasets:
        args.dataset = dataset
        if args.refresh_preprocessed:
            pre_dir = LRUREC_ROOT / 'data' / 'preprocessed' / f'{args.dataset}_min_rating0-min_uc5-min_sc5-leave_one_out'
            if pre_dir.exists():
                shutil.rmtree(pre_dir)
                print(f'[INFO] removed cached preprocessed dir: {pre_dir}')

        runs = []
        for seed in seeds:
            print(f'[RUN] lrurec_official @ {args.dataset} seed={seed}')
            runs.append(_run_one_seed(args, seed))

        hit = _mean_std([r['metrics']['hit@10'] for r in runs])
        ndcg = _mean_std([r['metrics']['ndcg@10'] for r in runs])
        mrr = _mean_std([r['metrics']['mrr@10'] for r in runs])
        all_results.append({
            'dataset': args.dataset,
            'epochs': args.epochs,
            'patience': args.patience,
            'max_seq_length': args.max_seq_length,
            'eval_mode': args.eval_mode,
            'num_neg': args.num_neg,
            'runs': runs,
            'summary': {
                'hit@10': hit,
                'ndcg@10': ndcg,
                'mrr@10': mrr,
            },
        })
        _print_summary('lrurec_official', args.dataset, all_results[-1]['summary'])

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
