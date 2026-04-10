# scripts/prepare_amazon_5core_subset.py
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Amazon 5-core subset for RecBole.")
    parser.add_argument("--input", required=True, help="Path to Amazon 5-core .json.gz file")
    parser.add_argument("--dataset-name", required=True, help="Dataset name for output folder/file")
    parser.add_argument("--min-user-inter", type=int, default=20, help="Min interactions per user")
    parser.add_argument("--min-item-inter", type=int, default=5, help="Min interactions per item")
    parser.add_argument("--max-users", type=int, default=0, help="Keep top-N users by interactions (0 = no limit)")
    return parser.parse_args()


def stream_reviews(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    args = parse_args()
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    user_cnt = Counter()
    item_cnt = Counter()
    for r in stream_reviews(src):
        user = r.get("reviewerID")
        item = r.get("asin")
        if user is None or item is None:
            continue
        user_cnt[user] += 1
        item_cnt[item] += 1

    users = {u for u, c in user_cnt.items() if c >= args.min_user_inter}
    if args.max_users and args.max_users > 0:
        users = set(u for u, _ in user_cnt.most_common(args.max_users) if u in users)

    items = {i for i, c in item_cnt.items() if c >= args.min_item_inter}

    out_dir = Path("data/amazon") / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset_name}.inter"

    kept_users = Counter()
    kept_items = Counter()
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for r in stream_reviews(src):
            user = r.get("reviewerID")
            item = r.get("asin")
            if user not in users or item not in items:
                continue
            rating = r.get("overall")
            ts = r.get("unixReviewTime")
            if rating is None or ts is None:
                continue
            f.write(f"{user}\t{item}\t{rating}\t{ts}\n")
            kept_users[user] += 1
            kept_items[item] += 1
            total += 1

    print(f"[INFO] saved: {out_path}")
    print(f"[INFO] interactions: {total}")
    print(f"[INFO] users: {len(kept_users)}, items: {len(kept_items)}")


if __name__ == "__main__":
    main()
