from preprocess.movielens import load_movielens
from preprocess.steam import load_steam
from preprocess.amazon import load_amazon_csv

from preprocess.base import (
    sort_by_user_and_time,
    build_user_sequences,
    Dataset
)

DATASET = "steam"  # 改成 "steam" 就行


def main():
    if DATASET == "movielens":
        df = load_movielens("../data/movielens/ratings.csv")
    elif DATASET == "steam":
        df = load_steam("../data/steam/steam-200k.csv")
    elif DATASET == "amazon_books":
        df = load_amazon_csv("../data/amazon/books_sample.csv", min_rating=None)
    elif DATASET == "amazon_elec":
        df = load_amazon_csv("../data/amazon/electronics_sample.csv", min_rating=None)
    else:
        raise ValueError("Unknown dataset")

    df = sort_by_user_and_time(df)
    user_seq, user_dt = build_user_sequences(df)
    dataset = Dataset(user_seq, user_dt, window_size=10)
    if len(dataset.seq_slices) == 0:
        print("[WARN] Empty dataset: no sequences generated.")
        return

    print(dataset.seq_slices[0])
    print(dataset.delta_t_slices[0])


if __name__ == "__main__":
    main()
