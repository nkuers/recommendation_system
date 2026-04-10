from pathlib import Path
import csv


SRC_DST = [
    (
        Path('data/movielens/ml-100k/ml-100k.inter'),
        Path('LRURec-master/data/ml100k_ours/ml100k_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-beauty-5core-subset/amazon-beauty-5core-subset.inter'),
        Path('LRURec-master/data/beauty_ours/beauty_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-electronics-5core-subset/amazon-electronics-5core-subset.inter'),
        Path('LRURec-master/data/electronics_ours/electronics_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-sports-5core-subset/amazon-sports-5core-subset.inter'),
        Path('LRURec-master/data/sports_ours/sports_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-subset/amazon-toys-5core-subset.inter'),
        Path('LRURec-master/data/toys_ours/toys_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-longseq/amazon-toys-5core-longseq.inter'),
        Path('LRURec-master/data/toyslong_ours/toyslong_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-subset-50k/amazon-toys-5core-subset-50k.inter'),
        Path('LRURec-master/data/toys50k_ours/toys50k_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-beauty-5core-longseq/amazon-beauty-5core-longseq.inter'),
        Path('LRURec-master/data/beautylong_ours/beautylong_ours.tsv'),
    ),
    (
        Path('data/amazon/amazon-sports-5core-longseq/amazon-sports-5core-longseq.inter'),
        Path('LRURec-master/data/sportslong_ours/sportslong_ours.tsv'),
    ),
]


def _normalize_row(row):
    out = {}
    for k, v in row.items():
        if k is None:
            continue
        out[k.split(':')[0]] = v
    return out


def convert(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f'source not found: {src}')

    rows = []
    with src.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for raw in reader:
            row = _normalize_row(raw)
            rows.append((row['user_id'], row['item_id'], row['timestamp']))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['uid', 'sid', 'timestamp'])
        writer.writerows(rows)

    print(f'[DONE] {dst}')
    print(f'  rows={len(rows)}')


if __name__ == '__main__':
    for src, dst in SRC_DST:
        convert(src, dst)
