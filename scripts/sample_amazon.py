import gzip
import json
import csv
from tqdm import tqdm


def sample_amazon_json(
    input_path,
    output_csv,
    max_interactions
):
    """
    从 Amazon 5-core json.gz 中顺序采样 max_interactions 条
    输出为 CSV：userId,itemId,timestamp,rating
    """

    with gzip.open(input_path, "rt", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)
        writer.writerow(["userId", "itemId", "timestamp", "rating"])

        count = 0
        for line in tqdm(fin, desc=f"Sampling {output_csv}"):
            obj = json.loads(line)

            writer.writerow([
                obj["reviewerID"],
                obj["asin"],
                obj["unixReviewTime"],
                obj["overall"]
            ])

            count += 1
            if count >= max_interactions:
                break

    print(f"[DONE] Saved {count} interactions to {output_csv}")
if __name__ == "__main__":
    # Books
    sample_amazon_json(
        input_path="../data/amazon/Books_5.json.gz",
        output_csv="../data/amazon/books_sample.csv",
        max_interactions=100_000
    )

    # Electronics
    sample_amazon_json(
        input_path="../data/amazon/Electronics_5.json.gz",
        output_csv="../data/amazon/electronics_sample.csv",
        max_interactions=50_000
    )
