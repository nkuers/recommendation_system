import gzip
import json
import csv
from collections import Counter
from tqdm import tqdm


def count_user_freq(input_path):
    """Pass1：统计每个 reviewerID 在文件中出现次数"""
    cnt = Counter()
    with gzip.open(input_path, "rt", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"Counting users in {input_path}"):
            obj = json.loads(line)
            uid = obj.get("reviewerID", None)
            if uid is not None:
                cnt[uid] += 1
    return cnt


def sample_amazon_json_filtered(
    input_path,
    output_csv,
    max_interactions,
    min_user_interactions=5,
    rating_min=None
):
    """
    从 Amazon 5-core json.gz 中采样 max_interactions 条
    过滤：只保留在全量文件中交互数 >= min_user_interactions 的用户
    可选：rating_min（例如 4.0）只保留高评分行为（一般先不加）
    输出 CSV：userId,itemId,timestamp,rating
    """
    # Pass1: user frequency
    user_cnt = count_user_freq(input_path)
    print(f"[INFO] unique users = {len(user_cnt)}")

    # Pass2: write filtered sample
    with gzip.open(input_path, "rt", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)
        writer.writerow(["userId", "itemId", "timestamp", "rating"])

        count = 0
        for line in tqdm(fin, desc=f"Sampling {output_csv}"):
            obj = json.loads(line)

            uid = obj.get("reviewerID", None)
            asin = obj.get("asin", None)
            ts = obj.get("unixReviewTime", None)
            rating = obj.get("overall", None)

            if uid is None or asin is None or ts is None or rating is None:
                continue

            # 用户最小交互数过滤（关键：避免 sample 中大量单次用户）
            if user_cnt.get(uid, 0) < min_user_interactions:
                continue

            # 可选评分过滤
            if rating_min is not None and float(rating) < float(rating_min):
                continue

            writer.writerow([uid, asin, int(ts), float(rating)])
            count += 1
            if count >= max_interactions:
                break

    print(f"[DONE] Saved {count} interactions to {output_csv} "
          f"(min_user_interactions={min_user_interactions}, rating_min={rating_min})")


if __name__ == "__main__":
    # Books：建议 200k
    sample_amazon_json_filtered(
        input_path="../data/amazon/Books_5.json.gz",
        output_csv="../data/amazon/books_sample.csv",
        max_interactions=200_000,
        min_user_interactions=5,
        rating_min=None
    )

    # Electronics：建议 100k
    sample_amazon_json_filtered(
        input_path="../data/amazon/Electronics_5.json.gz",
        output_csv="../data/amazon/electronics_sample.csv",
        max_interactions=100_000,
        min_user_interactions=5,
        rating_min=None
    )
