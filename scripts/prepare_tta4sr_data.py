from pathlib import Path
import pandas as pd

SRC_DST = [
    (
        Path('data/amazon/amazon-beauty-5core-subset/amazon-beauty-5core-subset.inter'),
        Path('TTA4SR-main/SASRec/data/Beauty_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-electronics-5core-subset/amazon-electronics-5core-subset.inter'),
        Path('TTA4SR-main/SASRec/data/Electronics_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-sports-5core-subset/amazon-sports-5core-subset.inter'),
        Path('TTA4SR-main/SASRec/data/Sports_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-subset/amazon-toys-5core-subset.inter'),
        Path('TTA4SR-main/SASRec/data/Toys_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-longseq/amazon-toys-5core-longseq.inter'),
        Path('TTA4SR-main/SASRec/data/ToysLong_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-toys-5core-subset-50k/amazon-toys-5core-subset-50k.inter'),
        Path('TTA4SR-main/SASRec/data/Toys50K_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-beauty-5core-longseq/amazon-beauty-5core-longseq.inter'),
        Path('TTA4SR-main/SASRec/data/BeautyLong_ours.txt'),
    ),
    (
        Path('data/amazon/amazon-sports-5core-longseq/amazon-sports-5core-longseq.inter'),
        Path('TTA4SR-main/SASRec/data/SportsLong_ours.txt'),
    ),
    (
        Path('data/movielens/ml-100k/ml-100k.inter'),
        Path('TTA4SR-main/SASRec/data/ML100K_ours.txt'),
    ),
]


def convert(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f'source not found: {src}')
    df = pd.read_csv(src, sep='\t')
    df.columns = [c.split(':')[0] for c in df.columns]
    req = {'user_id', 'item_id', 'timestamp'}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f'missing columns in {src}: {sorted(miss)}')

    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    user_map = {u: i + 1 for i, u in enumerate(df['user_id'].unique())}
    item_map = {it: i + 1 for i, it in enumerate(df['item_id'].unique())}

    df['u'] = df['user_id'].map(user_map)
    df['i'] = df['item_id'].map(item_map)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as f:
        for u, g in df.groupby('u', sort=False):
            seq = ' '.join(str(x) for x in g['i'].tolist())
            f.write(f'{u}\t{seq}\n')

    print(f'[DONE] {dst}')
    print(f'  users={len(user_map)} items={len(item_map)} inters={len(df)}')


if __name__ == '__main__':
    for src, dst in SRC_DST:
        convert(src, dst)
