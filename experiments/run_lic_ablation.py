import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiments.run_comparative_experiment import (
    set_seed,
    get_default_epochs,
    load_and_process_dataset,
    split_train_test,
    build_dt_bucket_slices,
    compute_dt_quantile_bins,
    evaluate_on_test_samples,
)
from preprocess.base import Dataset
from models.lic_v1 import SeqTrainDatasetLIC, collate_lic, LICv1, bpr_loss as bpr_loss_lic


class LICOnlyGSU(LICv1):
    """仅使用 GSU（长序列）进行打分"""
    def score(self,
              items_long_pad, dts_long_pad, mask_long,
              items_short_pad, dts_short_pad, mask_short,
              cand_items):
        u_gsu, _, tau_gsu = self.gsu(items_long_pad, dts_long_pad, mask_long, cand_items)
        q = self.item_emb(cand_items)
        s = (u_gsu * q).sum(dim=-1)
        lam = q.new_tensor(1.0)
        tau_esu = q.new_tensor(0.0)
        return s, lam, tau_gsu, tau_esu


class LICOnlyESU(LICv1):
    """仅使用 ESU（短序列）进行打分"""
    def score(self,
              items_long_pad, dts_long_pad, mask_long,
              items_short_pad, dts_short_pad, mask_short,
              cand_items):
        u_esu, _, tau_esu = self.esu(items_short_pad, dts_short_pad, mask_short, cand_items)
        q = self.item_emb(cand_items)
        s = (u_esu * q).sum(dim=-1)
        lam = q.new_tensor(0.0)
        tau_gsu = q.new_tensor(0.0)
        return s, lam, tau_gsu, tau_esu


def train_one_variant(model, train_loader, device, num_epochs, patience=5, min_delta=1e-4):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_loss = None
    no_improve = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            items_long_pad, dts_long_pad, mask_long, \
            items_short_pad, dts_short_pad, mask_short, \
            pos, neg = batch

            items_long_pad = items_long_pad.to(device)
            dts_long_pad = dts_long_pad.to(device)
            mask_long = mask_long.to(device)
            items_short_pad = items_short_pad.to(device)
            dts_short_pad = dts_short_pad.to(device)
            mask_short = mask_short.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_s, neg_s, _, _, _ = model(
                items_long_pad, dts_long_pad, mask_long,
                items_short_pad, dts_short_pad, mask_short,
                pos, neg
            )
            loss = bpr_loss_lic(pos_s, neg_s)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"[Epoch {epoch + 1}/{num_epochs}] loss={avg_loss:.4f}")

        if best_loss is None or avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch + 1 >= 5:
                print(f"[Early Stop] No improvement for {patience} epochs. Stop at epoch {epoch + 1}.")
                break


def main():
    parser = argparse.ArgumentParser(description="LIC ablation on MovieLens.")
    parser.add_argument("--dataset", default="movielens",
                        choices=["movielens", "amazon_books", "amazon_electronics", "steam"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--long_window", type=int, default=30)
    parser.add_argument("--short_len", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_name = args.dataset

    df_sorted, user_seq, user_dt, num_items = load_and_process_dataset(dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    effective_epochs = args.epochs if args.epochs is not None else get_default_epochs("lic_v1")

    user_hours_full = None
    train_user_seq, train_user_dt, train_user_hours, test_samples = split_train_test(
        user_seq,
        user_dt,
        user_hours_full,
        window_size=10,
        long_window=args.long_window,
        short_len=args.short_len
    )
    if len(train_user_seq) == 0:
        print("[WARN] Empty train set after split. Skip.")
        return

    dt_bins = None
    if dataset_name in {"amazon_books", "amazon_electronics"}:
        dt_bins = compute_dt_quantile_bins(train_user_dt)

    ds_long = Dataset(train_user_seq, train_user_dt, window_size=args.long_window)
    dt_bucket_slices_long = build_dt_bucket_slices(ds_long.delta_t_slices, bins=dt_bins)
    num_dt_buckets = max(max(x) for x in dt_bucket_slices_long) + 1

    train_data = SeqTrainDatasetLIC(
        seq_slices_long=ds_long.seq_slices,
        dt_bucket_slices_long=dt_bucket_slices_long,
        num_items=num_items,
        short_len=args.short_len,
        max_steps=args.max_steps
    )
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_lic
    )

    variants = ["only_gsu", "only_esu", "full_lic"]
    results = {}
    for name in variants:
        results[name] = {"runs": {"HR@10": [], "NDCG@10": [], "MRR": [], "AUC": []}}
        for rep in range(args.repeats):
            run_seed = args.seed + rep
            set_seed(run_seed)
            if name == "only_gsu":
                model = LICOnlyGSU(num_items, num_dt_buckets, emb_dim=64, dropout_p=0.1).to(device)
            elif name == "only_esu":
                model = LICOnlyESU(num_items, num_dt_buckets, emb_dim=64, dropout_p=0.1).to(device)
            else:
                model = LICv1(num_items, num_dt_buckets, emb_dim=64, dropout_p=0.1).to(device)

            print(f"\n{'='*60}")
            print(f"Variant={name}, Repeat={rep + 1}/{args.repeats}")
            print(f"{'='*60}")
            train_one_variant(model, train_loader, device, effective_epochs)
            hr_k, ndcg_k, mrr, auc = evaluate_on_test_samples(
                model,
                test_samples,
                device,
                k=10,
                num_negatives=100,
                model_type="lic_v1",
                num_items=num_items,
                window_size=10,
                long_window=args.long_window,
                short_len=args.short_len,
                dt_bins=dt_bins
            )
            print(f"[Evaluation] HR@10={hr_k:.4f}, NDCG@10={ndcg_k:.4f}, MRR={mrr:.4f}, AUC={auc:.4f}")
            results[name]["runs"]["HR@10"].append(hr_k)
            results[name]["runs"]["NDCG@10"].append(ndcg_k)
            results[name]["runs"]["MRR"].append(mrr)
            results[name]["runs"]["AUC"].append(auc)

        results[name]["mean"] = {k: mean(v) for k, v in results[name]["runs"].items()}
        results[name]["std"] = {k: (stdev(v) if len(v) > 1 else 0.0)
                                for k, v in results[name]["runs"].items()}

    output_path = args.output
    if not output_path:
        out_dir = ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"lic_ablation_{dataset_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
