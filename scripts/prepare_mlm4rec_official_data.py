from collections import defaultdict
from pathlib import Path
import csv


SRC_DST = [
    (
        Path('data/movielens/ml-100k/ml-100k.inter'),
        Path('MLM4Rec-master/data/ML100K_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-beauty-5core-subset/amazon-beauty-5core-subset.inter'),
        Path('MLM4Rec-master/data/Beauty_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-beauty-5core-longseq/amazon-beauty-5core-longseq.inter'),
        Path('MLM4Rec-master/data/BeautyLong_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-electronics-5core-subset/amazon-electronics-5core-subset.inter'),
        Path('MLM4Rec-master/data/Electronics_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-sports-5core-subset/amazon-sports-5core-subset.inter'),
        Path('MLM4Rec-master/data/Sports_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-sports-5core-longseq/amazon-sports-5core-longseq.inter'),
        Path('MLM4Rec-master/data/SportsLong_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-subset/amazon-toys-5core-subset.inter'),
        Path('MLM4Rec-master/data/Toys_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-longseq/amazon-toys-5core-longseq.inter'),
        Path('MLM4Rec-master/data/ToysLong_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-subset-50k/amazon-toys-5core-subset-50k.inter'),
        Path('MLM4Rec-master/data/Toys50K_ours.txt'),
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

    user_events = defaultdict(list)
    with src.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for raw in reader:
            row = _normalize_row(raw)
            uid = str(row['user_id'])
            iid = str(row['item_id'])
            ts = float(row['timestamp'])
            user_events[uid].append((ts, iid))

    def _user_key(u):
        try:
            return (0, int(u))
        except Exception:
            return (1, u)

    users = sorted(user_events.keys(), key=_user_key)
    user_map = {u: i + 1 for i, u in enumerate(users)}
    item_map = {}
    next_item = 1
    kept = []

    for u in users:
        events = sorted(user_events[u], key=lambda x: x[0])
        seq = []
        for _, iid in events:
            if iid not in item_map:
                item_map[iid] = next_item
                next_item += 1
            seq.append(item_map[iid])
        if len(seq) >= 3:
            kept.append((user_map[u], seq))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', encoding='utf-8') as f:
        for uid, seq in kept:
            f.write(f"{uid} {' '.join(str(x) for x in seq)}\n")

    print(f'[DONE] {dst}')
    print(f'  users={len(kept)} items={len(item_map)} inters={sum(len(s) for _, s in kept)}')


if __name__ == '__main__':
    for src, dst in SRC_DST:
        convert(src, dst)
