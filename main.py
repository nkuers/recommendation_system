import math
from collections import Counter
import pandas as pd

from preprocess.base import sort_by_user_and_time
from preprocess.amazon import load_amazon_csv

BINS = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

def dt_to_bucket(dt_seconds: int) -> int:
    if dt_seconds is None or dt_seconds < 0:
        dt_seconds = 0
    x = math.log1p(int(dt_seconds))
    for i in range(1, len(BINS)):
        if x <= BINS[i]:
            return i
    return len(BINS)

def summarize(name, dts):
    flat = [dt_to_bucket(x) for x in dts]
    cnt = Counter(flat)
    total = sum(cnt.values())
    print(f"\n==================== {name} (by user diff) ====================")
    print(f"[INFO] total dt tokens = {total}")
    for b, c in cnt.most_common(10):
        print(f"  bucket {b}: count={c}, ratio={c/total:.4f}")
    top = cnt.most_common(3)
    cover3 = sum(c for _, c in top) / total
    print(f"[COVER] Top-3 ratio = {cover3:.4f}")

def main():
    # Books
    df_b = load_amazon_csv("data/amazon/books_sample.csv", keep_rating=False)
    df_b = sort_by_user_and_time(df_b)
    df_b["dt"] = df_b.groupby("userId")["timestamp"].diff()
    dts_b = df_b["dt"].dropna().astype(int).tolist()
    summarize("amazon_books", dts_b)

    # Electronics
    df_e = load_amazon_csv("data/amazon/electronics_sample.csv", keep_rating=False)
    df_e = sort_by_user_and_time(df_e)
    df_e["dt"] = df_e.groupby("userId")["timestamp"].diff()
    dts_e = df_e["dt"].dropna().astype(int).tolist()
    summarize("amazon_electronics", dts_e)

if __name__ == "__main__":
    main()
