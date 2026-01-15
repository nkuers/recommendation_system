import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess.movielens import load_movielens
from preprocess.amazon import load_amazon_csv
from preprocess.steam import load_steam
from preprocess.base import sort_by_user_and_time, build_user_sequences, Dataset

# 导入所有模型和相关的Dataset/collate函数
from models.baseline_mfseq import (
    SeqTrainDataset,
    collate_pad,
    MeanSeqModel,
    bpr_loss as bpr_loss_mfseq
)
from models.dt_attn import (
    SeqTrainDatasetWithDT as SeqTrainDatasetWithDT_Attn,
    collate_pad_items_dt as collate_pad_items_dt_attn,
    DTAttentionModel,
    bpr_loss as bpr_loss_dt_attn
)
from models.dt_seq import (
    SeqTrainDatasetWithDT as SeqTrainDatasetWithDT_Seq,
    collate_pad_items_dt as collate_pad_items_dt_seq,
    MeanSeqDTModel,
    bpr_loss as bpr_loss_dt_seq
)
from models.hour_seq import (
    SeqTrainDatasetWithHour,
    collate_pad_items_hours,
    MeanSeqHourModel,
    bpr_loss as bpr_loss_hour
)
from models.interest_clock_simple import (
    SeqTrainDatasetWithDT as SeqTrainDatasetWithDT_IC,
    collate_pad_items_dt as collate_pad_items_dt_ic,
    SimpleInterestClock,
    bpr_loss as bpr_loss_ic
)
from models.lic_v1 import (
    SeqTrainDatasetLIC,
    collate_lic,
    LICv1,
    bpr_loss as bpr_loss_lic
)


# ======================================================
# 工具函数
# ======================================================

def get_num_items(user_seq: dict) -> int:
    """计算item数量"""
    mx = 0
    for _, items in user_seq.items():
        if items:
            mx = max(mx, max(items))
    return mx + 1


def dt_to_bucket(dt_seconds: int, bins=None) -> int:
    """
    将时间差（秒）转换为bucket ID
    bucket 0: padding
    bucket 1..N: 时间尺度桶
    """
    if dt_seconds is None or dt_seconds < 0:
        dt_seconds = 0
    x = math.log1p(dt_seconds)

    if bins is None:
        bins = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        for i in range(1, len(bins)):
            if x <= bins[i]:
                return i
        return len(bins)

    for i, thr in enumerate(bins, start=1):
        if x <= thr:
            return i
    return len(bins)


def build_dt_bucket_slices(delta_t_slices, bins=None):
    """将delta_t序列转换为bucket序列"""
    return [[dt_to_bucket(int(dt), bins=bins) for dt in dt_seq] for dt_seq in delta_t_slices]


def compute_dt_quantile_bins(user_dt: dict, num_bins: int = 8):
    """基于 log1p(Δt) 的分位数分桶阈值（用于 Amazon）"""
    values = []
    for seq in user_dt.values():
        values.extend(seq)
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
    quantiles = np.quantile(arr, np.linspace(1.0 / num_bins, 1.0, num_bins))
    if np.unique(quantiles).size < num_bins:
        return None
    return quantiles.tolist()


def extract_hours_from_df(df):
    """
    从DataFrame中提取小时信息
    hour = (timestamp // 3600) % 24
    """
    df = df.copy()
    df["hour"] = ((df["timestamp"] // 3600) % 24).astype(int)
    user_hours = df.groupby("userId")["hour"].apply(list).to_dict()
    return user_hours


# 默认训练轮数配置
def get_default_epochs(model_type: str) -> int:
    """
    根据模型复杂度设定默认训练轮数：
      - 基础模型（mfseq, dt_seq, hour）: 20-50，取中位 30
      - 时间感知模型（dt_attn, interest_clock）: 30-50，取中位 40
      - 复杂模型（lic_v1）: 50-100，取中位 80
    """
    base = {"mfseq", "dt_seq", "hour"}
    time_aware = {"dt_attn", "interest_clock"}
    if model_type in base:
        return 100
    if model_type in time_aware:
        return 100
    if model_type == "lic_v1":
        return 100
    return 100


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================
# 标准评估指标计算函数
# ======================================================

def calculate_hr_at_k(rank, k):
    """计算Hit Rate@K"""
    return 1.0 if rank <= k else 0.0


def calculate_ndcg_at_k(rank, k):
    """计算NDCG@K"""
    if rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def calculate_mrr(rank):
    """计算MRR (Mean Reciprocal Rank)"""
    return 1.0 / rank if rank > 0 else 0.0


def calculate_auc(pos_score, neg_scores):
    """
    计算AUC (Area Under Curve)
    使用简化的AUC计算：正样本分数高于负样本的比例
    """
    pos_score_val = pos_score.item() if isinstance(pos_score, torch.Tensor) else pos_score
    neg_scores_vals = neg_scores.cpu().numpy() if isinstance(neg_scores, torch.Tensor) else neg_scores
    
    # 计算有多少个负样本的分数低于正样本
    correct = (neg_scores_vals < pos_score_val).sum()
    ties = (neg_scores_vals == pos_score_val).sum()
    auc = (correct + 0.5 * ties) / len(neg_scores_vals) if len(neg_scores_vals) > 0 else 0.0
    
    return auc


def compute_item_scores(model, model_type, context_data, candidate_items, device):
    """
    计算候选item的分数
    context_data: 根据模型类型不同，包含不同的上下文信息
    candidate_items: [B, N] 或 [N] 候选item ID
    """
    if model_type == "mfseq":
        items_pad, mask = context_data
        items_pad = items_pad.to(device)
        mask = mask.to(device)
        h = model.encode(items_pad, mask)  # [B, D]
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)  # [1, N]
            h = h.unsqueeze(0) if h.dim() == 1 else h  # [1, D]
        candidate_e = model.item_emb(candidate_items.to(device))  # [B, N, D]
        scores = (h.unsqueeze(1) * candidate_e).sum(dim=-1)  # [B, N]
        return scores.squeeze(0) if scores.shape[0] == 1 else scores
        
    elif model_type == "dt_seq":
        items_pad, dts_pad, mask = context_data
        items_pad = items_pad.to(device)
        dts_pad = dts_pad.to(device)
        mask = mask.to(device)
        h = model.encode(items_pad, dts_pad, mask)  # [B, D]
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
            h = h.unsqueeze(0) if h.dim() == 1 else h
        candidate_e = model.item_emb(candidate_items.to(device))
        scores = (h.unsqueeze(1) * candidate_e).sum(dim=-1)
        return scores.squeeze(0) if scores.shape[0] == 1 else scores
        
    elif model_type == "hour":
        items_pad, hours_pad, mask = context_data
        items_pad = items_pad.to(device)
        hours_pad = hours_pad.to(device)
        mask = mask.to(device)
        h = model.encode(items_pad, hours_pad, mask)  # [B, D]
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
            h = h.unsqueeze(0) if h.dim() == 1 else h
        candidate_e = model.item_emb(candidate_items.to(device))
        scores = (h.unsqueeze(1) * candidate_e).sum(dim=-1)
        return scores.squeeze(0) if scores.shape[0] == 1 else scores
        
    elif model_type == "dt_attn":
        items_pad, dts_pad, mask = context_data
        items_pad = items_pad.to(device)
        dts_pad = dts_pad.to(device)
        mask = mask.to(device)
        # dt_attn的encode与候选item无关，可以向量化计算
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
        h, _ = model.encode(items_pad, dts_pad, mask)  # [B, D]
        candidate_e = model.item_emb(candidate_items.to(device))  # [B, N, D]
        scores = (h.unsqueeze(1) * candidate_e).sum(dim=-1)  # [B, N]
        return scores.squeeze(0) if scores.shape[0] == 1 else scores
        
    elif model_type == "interest_clock":
        items_pad, dts_pad, mask = context_data
        items_pad = items_pad.to(device)
        dts_pad = dts_pad.to(device)
        mask = mask.to(device)
        # interest_clock也需要候选item，逐个计算
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
        scores = []
        for i in range(candidate_items.shape[1]):
            cand = candidate_items[:, i]
            u, _ = model._candidate_aware_uservec(items_pad, dts_pad, mask, cand.to(device))
            q = model.item_emb(cand.to(device))
            score = (u * q).sum(dim=-1)
            scores.append(score)
        scores = torch.stack(scores, dim=1)
        return scores.squeeze(0) if scores.shape[0] == 1 else scores
        
    elif model_type == "lic_v1":
        items_long_pad, dts_long_pad, mask_long, items_short_pad, dts_short_pad, mask_short = context_data
        items_long_pad = items_long_pad.to(device)
        dts_long_pad = dts_long_pad.to(device)
        mask_long = mask_long.to(device)
        items_short_pad = items_short_pad.to(device)
        dts_short_pad = dts_short_pad.to(device)
        mask_short = mask_short.to(device)
        # lic_v1需要候选item，逐个计算
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
        scores = []
        for i in range(candidate_items.shape[1]):
            cand = candidate_items[:, i]
            score, _, _, _ = model.score(
                items_long_pad, dts_long_pad, mask_long,
                items_short_pad, dts_short_pad, mask_short,
                cand.to(device)
            )
            scores.append(score)
        scores = torch.stack(scores, dim=1)
        return scores.squeeze(0) if scores.shape[0] == 1 else scores
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def split_train_test(user_seq: dict,
                     user_dt: dict,
                     user_hours: dict | None,
                     window_size: int = 10,
                     long_window: int = 30,
                     short_len: int = 10,
                     filter_cold_start: bool = True):
    """
    按用户拆分 train/test，并准备测试上下文：
      train = items[:-1], test_pos = items[-1]
      context = train 的最后 window_size 个
      评估时仅用 test_pos；负样本需排除用户历史
    返回：
      train_user_seq, train_user_dt, train_user_hours, test_samples(list[dict])
    """
    train_user_seq = {}
    train_user_dt = {}
    train_user_hours = {} if user_hours is not None else None
    train_item_set = set()
    tmp_samples = []
    test_samples = []

    for uid, seq in user_seq.items():
        dt_seq = user_dt[uid]
        if len(seq) < 2:
            continue
        train_seq = seq[:-1]
        train_dt = dt_seq[:-1]
        test_pos = seq[-1]
        history_set = set(train_seq)

        # 训练字典
        train_user_seq[uid] = train_seq
        train_user_dt[uid] = train_dt
        train_item_set.update(train_seq)
        if user_hours is not None:
            hrs = user_hours[uid]
            train_user_hours[uid] = hrs[:-1]
            context_hours = train_user_hours[uid][-window_size:]
        else:
            context_hours = None

        # 测试上下文
        context_items = train_seq[-window_size:]
        context_dt = train_dt[-window_size:]

        # LIC 需要 long/short
        long_items = train_seq[-long_window:]
        long_dt = train_dt[-long_window:]
        short_items = train_seq[-short_len:]
        short_dt = train_dt[-short_len:]

        tmp_samples.append({
            "uid": uid,
            "test_pos": test_pos,
            "history_set": history_set,
            "context_items": context_items,
            "context_dt": context_dt,
            "context_hours": context_hours,
            "long_items": long_items,
            "long_dt": long_dt,
            "short_items": short_items,
            "short_dt": short_dt,
        })

    for sample in tmp_samples:
        if filter_cold_start and sample["test_pos"] not in train_item_set:
            continue
        test_samples.append(sample)

    return train_user_seq, train_user_dt, train_user_hours, test_samples


def pad_context(items, dts=None, hours=None, pad_item_id=0, pad_dt_id=0, pad_hour_id=24, max_len=None):
    orig_len = len(items)
    if max_len is not None:
        if orig_len > max_len:
            items = items[-max_len:]
            if dts is not None:
                dts = dts[-max_len:]
            if hours is not None:
                hours = hours[-max_len:]
            orig_len = max_len
        pad_len = max_len - orig_len
        if pad_len > 0:
            items = items + [pad_item_id] * pad_len
            if dts is not None:
                dts = dts + [pad_dt_id] * pad_len
            if hours is not None:
                hours = hours + [pad_hour_id] * pad_len
    items_pad = torch.tensor(items, dtype=torch.long).unsqueeze(0)  # [1, L]
    mask = torch.zeros((1, len(items)), dtype=torch.bool)
    mask[0, :orig_len] = True
    dts_pad = None
    hours_pad = None
    if dts is not None:
        dts_pad = torch.tensor(dts, dtype=torch.long).unsqueeze(0)
    if hours is not None:
        hours_pad = torch.tensor(hours, dtype=torch.long).unsqueeze(0)
    return items_pad, dts_pad, hours_pad, mask


def evaluate_on_test_samples(model,
                             test_samples,
                             device,
                             k=10,
                             num_negatives=100,
                             model_type="default",
                             num_items=None,
                             window_size=10,
                             long_window=30,
                             short_len=10,
                             dt_bins=None):
    """
    仅在每个用户的 test_pos 上评估，负样本排除用户历史
    """
    import random

    model.eval()
    hr_k_sum = 0.0
    ndcg_k_sum = 0.0
    mrr_sum = 0.0
    auc_sum = 0.0
    total_samples = 0

    all_items_set = set(range(1, num_items)) if num_items is not None else None

    with torch.no_grad():
        for sample in test_samples:
            test_pos = sample["test_pos"]
            history_set = sample["history_set"]
            context_items = sample["context_items"]
            context_dt = sample["context_dt"]
            context_hours = sample["context_hours"]
            long_items = sample["long_items"]
            long_dt = sample["long_dt"]
            short_items = sample["short_items"]
            short_dt = sample["short_dt"]

            # 负样本采样：排除 test_pos 和历史
            if all_items_set is None:
                raise ValueError("num_items is required for negative sampling")
            candidates = list(all_items_set - history_set - {test_pos})
            if len(candidates) == 0:
                continue
            if len(candidates) >= num_negatives:
                neg_candidates = random.sample(candidates, num_negatives)
            else:
                neg_candidates = candidates

            all_candidates = torch.tensor([test_pos] + neg_candidates, dtype=torch.long)

            if model_type == "mfseq":
                items_pad, _, _, mask = pad_context(context_items, max_len=window_size)
                context_data = (items_pad, mask)
            elif model_type in ["dt_attn", "dt_seq", "interest_clock"]:
                dt_buckets = [dt_to_bucket(int(dt), bins=dt_bins) for dt in context_dt]
                items_pad, dts_pad, _, mask = pad_context(context_items, dt_buckets, max_len=window_size)
                context_data = (items_pad, dts_pad, mask)
            elif model_type == "hour":
                dt_dummy = None
                items_pad, _, hours_pad, mask = pad_context(context_items, dt_dummy, context_hours, max_len=window_size)
                context_data = (items_pad, hours_pad, mask)
            elif model_type == "lic_v1":
                long_dt_buckets = [dt_to_bucket(int(dt), bins=dt_bins) for dt in long_dt]
                short_dt_buckets = [dt_to_bucket(int(dt), bins=dt_bins) for dt in short_dt]
                items_long_pad, dts_long_pad, _, mask_long = pad_context(
                    long_items, long_dt_buckets, max_len=long_window
                )
                items_short_pad, dts_short_pad, _, mask_short = pad_context(
                    short_items, short_dt_buckets, max_len=short_len
                )
                context_data = (
                    items_long_pad, dts_long_pad, mask_long,
                    items_short_pad, dts_short_pad, mask_short
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            all_scores = compute_item_scores(model, model_type, context_data, all_candidates, device)
            _, sorted_indices = torch.sort(all_scores, descending=True)
            rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1

            hr_k_sum += calculate_hr_at_k(rank, k)
            ndcg_k_sum += calculate_ndcg_at_k(rank, k)
            mrr_sum += calculate_mrr(rank)
            pos_score = all_scores[0]
            neg_scores = all_scores[1:]
            auc_sum += calculate_auc(pos_score, neg_scores)
            total_samples += 1

    hr_k = hr_k_sum / total_samples if total_samples > 0 else 0
    ndcg_k = ndcg_k_sum / total_samples if total_samples > 0 else 0
    mrr = mrr_sum / total_samples if total_samples > 0 else 0
    auc = auc_sum / total_samples if total_samples > 0 else 0
    return hr_k, ndcg_k, mrr, auc


# ======================================================
# 数据集加载和预处理
# ======================================================

def load_and_process_dataset(dataset_name):
    """加载数据集并进行预处理"""
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    # 加载数据
    if dataset_name == "movielens":
        df = load_movielens(str(data_dir / "movielens" / "ratings.csv"))
    elif dataset_name == "amazon_books":
        df = load_amazon_csv(str(data_dir / "amazon" / "books_sample.csv"))
    elif dataset_name == "amazon_electronics":
        df = load_amazon_csv(str(data_dir / "amazon" / "electronics_sample.csv"))
    elif dataset_name == "steam":
        df = load_steam(str(data_dir / "steam" / "steam-200k.csv"))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 排序
    df_sorted = sort_by_user_and_time(df)
    
    # 构建序列
    user_seq, user_dt = build_user_sequences(df_sorted)
    
    # 计算num_items
    num_items = get_num_items(user_seq)
    
    return df_sorted, user_seq, user_dt, num_items


# ======================================================
# 训练和评估函数
# ======================================================

def train_and_evaluate(dataset_name, model_type, num_epochs=None, window_size=10, patience=5, min_delta=1e-4):
    """训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"Training {model_type} on {dataset_name}")
    print(f"{'='*60}")
    
    # 1. 加载数据集
    df_sorted, user_seq, user_dt, num_items = load_and_process_dataset(dataset_name)
    print(f"[INFO] num_items = {num_items}")
    
    # 2. 根据模型类型准备数据
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 确定训练轮数
    effective_epochs = num_epochs if num_epochs is not None else get_default_epochs(model_type)
    print(f"[INFO] Training epochs set to {effective_epochs}")

    # 轻量正则/降复杂度：仅针对 MovieLens / Amazon Books
    if dataset_name in {"movielens", "amazon_books"}:
        reg_dropout = 0.1
        lic_long_window = 30
    else:
        reg_dropout = 0.0
        lic_long_window = 30
    
    # 0) 按用户划分 train/test
    user_hours_full = extract_hours_from_df(df_sorted) if model_type == "hour" else None
    train_user_seq, train_user_dt, train_user_hours, test_samples = split_train_test(
        user_seq,
        user_dt,
        user_hours_full,
        window_size=window_size,
        long_window=lic_long_window,
        short_len=10
    )
    dt_bins = None
    if dataset_name in {"amazon_books", "amazon_electronics"}:
        dt_bins = compute_dt_quantile_bins(train_user_dt)

    if len(train_user_seq) == 0:
        print("[WARN] Empty train set after split. Skip.")
        return 0, 0, 0, 0

    if model_type == "mfseq":
        # 基础模型：只需要items
        ds = Dataset(train_user_seq, train_user_dt, window_size=window_size)
        train_data = SeqTrainDataset(ds.seq_slices, num_items=num_items)
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_pad
        )
        model = MeanSeqModel(num_items, emb_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    elif model_type == "dt_attn":
        # DT Attention模型：需要items和dt_buckets
        ds = Dataset(train_user_seq, train_user_dt, window_size=window_size)
        dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices, bins=dt_bins)
        num_dt_buckets = max(max(x) for x in dt_bucket_slices) + 1
        print(f"[INFO] num_dt_buckets = {num_dt_buckets}")
        
        train_data = SeqTrainDatasetWithDT_Attn(ds.seq_slices, dt_bucket_slices, num_items=num_items)
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_pad_items_dt_attn
        )
        model = DTAttentionModel(num_items, num_dt_buckets, emb_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    elif model_type == "dt_seq":
        # DT Sequence模型：需要items和dt_buckets
        ds = Dataset(train_user_seq, train_user_dt, window_size=window_size)
        dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices, bins=dt_bins)
        num_dt_buckets = max(max(x) for x in dt_bucket_slices) + 1
        print(f"[INFO] num_dt_buckets = {num_dt_buckets}")
        
        train_data = SeqTrainDatasetWithDT_Seq(ds.seq_slices, dt_bucket_slices, num_items=num_items)
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_pad_items_dt_seq
        )
        model = MeanSeqDTModel(num_items, num_dt_buckets, emb_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    elif model_type == "hour":
        # Hour模型：需要items和hours
        user_hours = train_user_hours
        ds = Dataset(train_user_seq, train_user_dt, window_size=window_size, user_hours=user_hours)
        hour_slices = ds.hour_slices
        
        train_data = SeqTrainDatasetWithHour(ds.seq_slices, hour_slices, num_items=num_items)
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_pad_items_hours
        )
        model = MeanSeqHourModel(num_items, emb_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    elif model_type == "interest_clock":
        # Interest Clock模型：需要items和dt_buckets
        ds = Dataset(train_user_seq, train_user_dt, window_size=window_size)
        dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices, bins=dt_bins)
        num_dt_buckets = max(max(x) for x in dt_bucket_slices) + 1
        print(f"[INFO] num_dt_buckets = {num_dt_buckets}")
        
        train_data = SeqTrainDatasetWithDT_IC(ds.seq_slices, dt_bucket_slices, num_items=num_items)
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_pad_items_dt_ic
        )
        model = SimpleInterestClock(num_items, num_dt_buckets, emb_dim=64, dropout_p=reg_dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        
    elif model_type == "lic_v1":
        # LIC模型：需要长窗口和短窗口
        long_window = lic_long_window
        short_len = 10
        ds_long = Dataset(train_user_seq, train_user_dt, window_size=long_window)
        dt_bucket_slices_long = build_dt_bucket_slices(ds_long.delta_t_slices, bins=dt_bins)
        num_dt_buckets = max(max(x) for x in dt_bucket_slices_long) + 1
        print(f"[INFO] num_dt_buckets = {num_dt_buckets}")
        print(f"[INFO] long_window = {long_window}, short_len = {short_len}")
        
        train_data = SeqTrainDatasetLIC(
            seq_slices_long=ds_long.seq_slices,
            dt_bucket_slices_long=dt_bucket_slices_long,
            num_items=num_items,
            short_len=short_len,
            max_steps=10
        )
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_lic
        )
        model = LICv1(num_items, num_dt_buckets, emb_dim=64, dropout_p=reg_dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # 3. 训练
    print(f"[INFO] Starting training for {effective_epochs} epochs...")
    model.train()
    best_loss = None
    no_improve = 0

    for epoch in range(effective_epochs):
        total_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 根据模型类型处理不同的batch格式
            if model_type == "mfseq":
                items_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                pos_s, neg_s = model(items_pad, mask, pos, neg)
                loss = bpr_loss_mfseq(pos_s, neg_s)
                
            elif model_type == "dt_attn":
                items_pad, dts_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                dts_pad = dts_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                pos_s, neg_s, _ = model(items_pad, dts_pad, mask, pos, neg)
                loss = bpr_loss_dt_attn(pos_s, neg_s)
                
            elif model_type == "dt_seq":
                items_pad, dts_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                dts_pad = dts_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                pos_s, neg_s = model(items_pad, dts_pad, mask, pos, neg)
                loss = bpr_loss_dt_seq(pos_s, neg_s)
                
            elif model_type == "hour":
                items_pad, hours_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                hours_pad = hours_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                pos_s, neg_s = model(items_pad, hours_pad, mask, pos, neg)
                loss = bpr_loss_hour(pos_s, neg_s)
                
            elif model_type == "interest_clock":
                items_pad, dts_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                dts_pad = dts_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                pos_s, neg_s, _ = model(items_pad, dts_pad, mask, pos, neg)
                loss = bpr_loss_ic(pos_s, neg_s)
                
            elif model_type == "lic_v1":
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
        print(f"[Epoch {epoch+1}/{effective_epochs}] loss={avg_loss:.4f}")

        # 简单早停：若连续 patience 轮无显著提升则停止
        if best_loss is None or avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch + 1 >= 5:  # 至少训练若干轮再早停
                print(f"[Early Stop] No improvement for {patience} epochs. Stop at epoch {epoch+1}.")
                break
    
    # 4. 评估
    print(f"[INFO] Evaluating model...")
    hr_k, ndcg_k, mrr, auc = evaluate_on_test_samples(
        model,
        test_samples,
        device,
        k=10,
        num_negatives=100,
        model_type=model_type,
        num_items=num_items,
        window_size=window_size,
        long_window=lic_long_window,
        short_len=10,
        dt_bins=dt_bins
    )
    print(f"[Evaluation] HR@10={hr_k:.4f}, NDCG@10={ndcg_k:.4f}, MRR={mrr:.4f}, AUC={auc:.4f}")
    
    return hr_k, ndcg_k, mrr, auc


# ======================================================
# 对比实验主函数
# ======================================================

def run_comparative_experiment():
    """运行所有模型在所有数据集上的对比实验"""
    set_seed(42)
    model_types = ["mfseq", "dt_attn", "dt_seq", "hour", "interest_clock", "lic_v1"]
    datasets = ["movielens", "amazon_books", "amazon_electronics", "steam"]
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*60}")
        results[dataset_name] = {}
        
        for model_type in model_types:
            try:
                hr_k, ndcg_k, mrr, auc = train_and_evaluate(
                    dataset_name, 
                    model_type, 
                    num_epochs=None,  # 使用模型类型默认轮数
                    window_size=10
                )
                results[dataset_name][model_type] = {
                    "HR@10": hr_k,
                    "NDCG@10": ndcg_k,
                    "MRR": mrr,
                    "AUC": auc
                }
            except Exception as e:
                print(f"[ERROR] Failed to train {model_type} on {dataset_name}: {e}")
                results[dataset_name][model_type] = {"error": str(e)}
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        for model_type in model_types:
            if model_type in results[dataset_name]:
                result = results[dataset_name][model_type]
                if "error" in result:
                    print(f"  {model_type}: ERROR - {result['error']}")
                else:
                    print(f"  {model_type}: HR@10={result['HR@10']:.4f}, "
                          f"NDCG@10={result['NDCG@10']:.4f}, "
                          f"MRR={result['MRR']:.4f}, "
                          f"AUC={result['AUC']:.4f}")


if __name__ == "__main__":
    run_comparative_experiment()
