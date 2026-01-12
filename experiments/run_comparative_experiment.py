import math
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


def dt_to_bucket(dt_seconds: int) -> int:
    """
    将时间差（秒）转换为bucket ID
    bucket 0: padding
    bucket 1..N: 时间尺度桶
    """
    if dt_seconds is None or dt_seconds < 0:
        dt_seconds = 0
    x = math.log1p(dt_seconds)
    
    bins = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for i in range(1, len(bins)):
        if x <= bins[i]:
            return i
    return len(bins)


def build_dt_bucket_slices(delta_t_slices):
    """将delta_t序列转换为bucket序列"""
    return [[dt_to_bucket(int(dt)) for dt in dt_seq] for dt_seq in delta_t_slices]


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
        return 30
    if model_type in time_aware:
        return 40
    if model_type == "lic_v1":
        return 80
    return 30


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
    auc = correct / len(neg_scores_vals) if len(neg_scores_vals) > 0 else 0.0
    
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
        # dt_attn的encode方法需要候选item，所以需要逐个计算
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
        scores = []
        for i in range(candidate_items.shape[1]):
            cand = candidate_items[:, i]  # [B]
            # dt_attn的forward需要pos和neg，但我们可以用encode方法
            # 实际上dt_attn的encode返回(h, attn)，h是用户表示
            # 但dt_attn的encode需要items_pad, dts_pad, mask，不依赖候选item
            # 所以我们可以直接计算
            h, _ = model.encode(items_pad, dts_pad, mask)  # [B, D]
            cand_e = model.item_emb(cand.to(device))  # [B, D]
            score = (h * cand_e).sum(dim=-1)  # [B]
            scores.append(score)
        scores = torch.stack(scores, dim=1)  # [B, N]
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


def evaluate_model(model, data_loader, device, k=10, num_negatives=100, model_type="default", num_items=None):
    """
    标准评估模型性能
    为每个正样本生成多个负样本，计算HR@K、NDCG@K、MRR和AUC
    
    参数:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        k: Top-K值
        num_negatives: 负样本数量
        model_type: 模型类型
        num_items: item总数（用于生成负样本）
    """
    import random
    
    model.eval()
    total_samples = 0
    
    hr_k_sum = 0.0
    ndcg_k_sum = 0.0
    mrr_sum = 0.0
    auc_sum = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch_size = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
            
            # 根据模型类型处理不同的batch格式
            if model_type == "mfseq":
                items_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                
                # 计算正样本分数
                pos_s, _ = model(items_pad, mask, pos, neg)  # 只需要正样本分数
                bpr_loss_fn = bpr_loss_mfseq
                
            elif model_type in ["dt_attn", "dt_seq", "interest_clock"]:
                items_pad, dts_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                dts_pad = dts_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                
                if model_type == "dt_attn":
                    pos_s, _, _ = model(items_pad, dts_pad, mask, pos, neg)
                    bpr_loss_fn = bpr_loss_dt_attn
                elif model_type == "dt_seq":
                    pos_s, _ = model(items_pad, dts_pad, mask, pos, neg)
                    bpr_loss_fn = bpr_loss_dt_seq
                elif model_type == "interest_clock":
                    pos_s, _, _ = model(items_pad, dts_pad, mask, pos, neg)
                    bpr_loss_fn = bpr_loss_ic
                    
            elif model_type == "hour":
                items_pad, hours_pad, mask, pos, neg = batch
                items_pad = items_pad.to(device)
                hours_pad = hours_pad.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                pos_s, _ = model(items_pad, hours_pad, mask, pos, neg)
                bpr_loss_fn = bpr_loss_hour
                
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
                
                pos_s, _, _, _, _ = model(
                    items_long_pad, dts_long_pad, mask_long,
                    items_short_pad, dts_short_pad, mask_short,
                    pos, neg
                )
                bpr_loss_fn = bpr_loss_lic
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            # 为每个样本生成多个负样本并计算分数
            for i in range(batch_size):
                pos_item = pos[i].item()
                
                # 生成负样本（排除正样本）
                if num_items is not None:
                    neg_candidates = []
                    while len(neg_candidates) < num_negatives:
                        candidate = random.randint(0, num_items - 1)
                        if candidate != pos_item:
                            neg_candidates.append(candidate)
                    
                    # 合并正样本和负样本
                    all_candidates = torch.tensor([pos_item] + neg_candidates, dtype=torch.long)  # [1 + num_negatives]
                    
                    # 准备上下文数据
                    if model_type == "mfseq":
                        context_data = (items_pad[i:i+1], mask[i:i+1])
                    elif model_type in ["dt_attn", "dt_seq", "interest_clock"]:
                        context_data = (items_pad[i:i+1], dts_pad[i:i+1], mask[i:i+1])
                    elif model_type == "hour":
                        context_data = (items_pad[i:i+1], hours_pad[i:i+1], mask[i:i+1])
                    elif model_type == "lic_v1":
                        context_data = (
                            items_long_pad[i:i+1], dts_long_pad[i:i+1], mask_long[i:i+1],
                            items_short_pad[i:i+1], dts_short_pad[i:i+1], mask_short[i:i+1]
                        )
                    
                    # 计算所有候选item的分数
                    all_scores = compute_item_scores(model, model_type, context_data, all_candidates, device)
                    
                    # 排序并找到正样本的排名（从1开始）
                    _, sorted_indices = torch.sort(all_scores, descending=True)
                    rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
                    
                    # 计算指标
                    hr_k_sum += calculate_hr_at_k(rank, k)
                    ndcg_k_sum += calculate_ndcg_at_k(rank, k)
                    mrr_sum += calculate_mrr(rank)
                    
                    # 计算AUC：正样本分数高于负样本的比例
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
    # 加载数据
    if dataset_name == "movielens":
        df = load_movielens("../data/movielens/ratings.csv")
    elif dataset_name == "amazon_books":
        df = load_amazon_csv("../data/amazon/books_sample.csv")
    elif dataset_name == "amazon_electronics":
        df = load_amazon_csv("../data/amazon/electronics_sample.csv")
    elif dataset_name == "steam":
        df = load_steam("../data/steam/steam-200k.csv")
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
    
    if model_type == "mfseq":
        # 基础模型：只需要items
        ds = Dataset(user_seq, user_dt, window_size=window_size)
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
        ds = Dataset(user_seq, user_dt, window_size=window_size)
        dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices)
        num_dt_buckets = max(max(x) for x in dt_bucket_slices) + 1
        print(f"[INFO] num_dt_buckets = {num_dt_buckets}")
        
        train_data = SeqTrainDatasetWithDT_Attn(ds.seq_slices, dt_bucket_slices, num_items=num_items)
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_pad_items_dt_attn
        )
        model = DTAttentionModel(num_items, num_dt_buckets, emb_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    elif model_type == "dt_seq":
        # DT Sequence模型：需要items和dt_buckets
        ds = Dataset(user_seq, user_dt, window_size=window_size)
        dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices)
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
        user_hours = extract_hours_from_df(df_sorted)
        ds = Dataset(user_seq, user_dt, window_size=window_size, user_hours=user_hours)
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
        ds = Dataset(user_seq, user_dt, window_size=window_size)
        dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices)
        num_dt_buckets = max(max(x) for x in dt_bucket_slices) + 1
        print(f"[INFO] num_dt_buckets = {num_dt_buckets}")
        
        train_data = SeqTrainDatasetWithDT_IC(ds.seq_slices, dt_bucket_slices, num_items=num_items)
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_pad_items_dt_ic
        )
        model = SimpleInterestClock(num_items, num_dt_buckets, emb_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    elif model_type == "lic_v1":
        # LIC模型：需要长窗口和短窗口
        long_window = 50
        short_len = 10
        ds_long = Dataset(user_seq, user_dt, window_size=long_window)
        dt_bucket_slices_long = build_dt_bucket_slices(ds_long.delta_t_slices)
        num_dt_buckets = max(max(x) for x in dt_bucket_slices_long) + 1
        print(f"[INFO] num_dt_buckets = {num_dt_buckets}")
        print(f"[INFO] long_window = {long_window}, short_len = {short_len}")
        
        train_data = SeqTrainDatasetLIC(
            seq_slices_long=ds_long.seq_slices,
            dt_bucket_slices_long=dt_bucket_slices_long,
            num_items=num_items,
            short_len=short_len
        )
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_lic
        )
        model = LICv1(num_items, num_dt_buckets, emb_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
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
    hr_k, ndcg_k, mrr, auc = evaluate_model(model, train_loader, device, k=10, num_negatives=100, model_type=model_type, num_items=num_items)
    print(f"[Evaluation] HR@10={hr_k:.4f}, NDCG@10={ndcg_k:.4f}, MRR={mrr:.4f}, AUC={auc:.4f}")
    
    return hr_k, ndcg_k, mrr, auc


# ======================================================
# 对比实验主函数
# ======================================================

def run_comparative_experiment():
    """运行所有模型在所有数据集上的对比实验"""
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
