import json
from itertools import product
from pathlib import Path

from recbole.quick_start import run_recbole

DATASETS = {
    "amazon-beauty-5core-subset": "recbole_config/amazon-beauty-5core-subset-idurl.yaml",
    "amazon-electronics-5core-subset": "recbole_config/amazon-electronics-5core-subset-idurl.yaml",
}
MODEL_CFG = "recbole_config/sasrec_idurl.yaml"
SEEDS = [42]

# Small grid: 2*2*3 = 12 configs
GRID = list(
    product(
        [5, 8],          # n_facet_all
        [0.1, 0.3],      # disen_lambda
        [0.2, 0.4, 0.6], # align_lambda
        [0.0005],        # learning_rate
        [0.2],           # hidden_dropout_prob
    )
)


def mean_metric(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main():
    out = {}
    for dataset, dataset_cfg in DATASETS.items():
        for facet, disen, align, lr, drop in GRID:
            tag = f"f{facet}_d{disen}_a{align}_lr{lr}_dp{drop}"
            raw = []
            for seed in SEEDS:
                cfg = {
                    "seed": seed,
                    "device": "gpu",
                    "epochs": 100,  # set to 30 for a quick pre-screen
                    "train_neg_sample_args": None,
                    "neg_sampling": None,
                    "n_facet_all": facet,
                    "disen_lambda": disen,
                    "align_lambda": align,
                    "learning_rate": lr,
                    "hidden_dropout_prob": drop,
                    "idra": 1,
                }
                result = run_recbole(
                    model="SASRec_IDURL",
                    dataset=dataset,
                    config_file_list=[dataset_cfg, MODEL_CFG],
                    config_dict=cfg,
                )
                raw.append(result["test_result"])

            m = {
                "hit@10": mean_metric(raw, "hit@10"),
                "ndcg@10": mean_metric(raw, "ndcg@10"),
                "mrr@10": mean_metric(raw, "mrr@10"),
            }
            key = f"{dataset}::{tag}"
            out[key] = {"mean": m, "raw": raw}
            print(f"{key} -> {m}")

    Path("results").mkdir(parents=True, exist_ok=True)
    save_path = Path("results/idurl_small_grid.json")
    save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[DONE] saved to {save_path}")


if __name__ == "__main__":
    main()
