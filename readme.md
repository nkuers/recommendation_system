## 0. 已完成的地基（你现在已经做到这里）

**数据层统一**（MovieLens/Steam/Amazon）
 统一输出：

- `seq_slices`: item 序列窗口
- `delta_t_slices`: 相邻行为时间差窗口（Steam 是伪时间步长）
- `hour_slices`: 小时窗口（MovieLens/Amazon 有；Steam 可选用伪）

**训练链路跑通**（BPR loss 只是过程监控）：

- Base baseline ✅
- Hour Embedding + 门控 α（softplus）✅
- Δt Embedding + 门控 β（softplus）✅

------

## 1. 模型对比实验的“全流程路线图”

你要做的模型按层级可以分成 4 组：

### A. 基线组（必须有）

1. **Base Model（无时间）**

- 输入：items
- 用户表示：mean pooling / GRU / SASRec（你现在用 mean pooling）
- 目的：作为所有时间模型的参照

------

### B. 传统时间建模组（任务书点名）

1. **Hour Embedding**

- 输入：items + hour
- 方式：`item_emb + α * hour_emb`
- 目的：对比传统 hour embedding 的效果与稳定性
- 你已完成（并做了门控轻量化）

------

### C. 连续时间间隔组（为 LIC 铺垫）

1. **Δt Embedding（简化 s(Δt)）**

- 输入：items + Δt（分桶或 log 压缩）
- 方式：`item_emb + β * dt_emb`
- 目的：证明“仅用时间差”是否足够
- 你已完成（分桶 + softplus β）

1. **Δt Weighting（时间衰减/时间权重）**

	models/dt_attn.py； experiments/train_dt_attn_baseline.py

- 输入：items + Δt
- 方式：不把 Δt 当 embedding，而是当权重
	 `h = Σ softmax(w(Δt_i)) * item_emb_i`
- w(Δt) 可以是：
	- 可学习的分桶权重（最稳）
	- 指数衰减 `exp(-λΔt)`（最轻量）
- 目的：把“时间差→注意力/权重”这条路跑通
- 这一步是 LIC 的“核心味道”，但比 LIC 简单

------

### D. 目标模型组（Interest Clock / LIC）

1. **Interest Clock（SIGIR 2024 思路）**

- 输入：items + Δt（相对候选 item 或当前时刻）
- 方式：时间差注意力 + 序列聚合
- 输出：用户向量对候选 item 自适应

1. **LIC：Long-Term Interest Clock（WWW 2025）**

- 你任务书里点名的两大模块：
	- **Clock-GSU**：从长期历史里“泛搜索”找相关兴趣
	- **Clock-ESU**：从精确历史里“精搜索”找匹配兴趣
- 共通核心：**候选 item 与历史序列之间用 Δt 驱动的注意力/匹配**
- 目的：在多数据集上证明 LIC 比 hour embedding / dt embedding 更强

------

## 2. 每个阶段你“输出物”是什么（老师最关心）

你可以按下面这个 checklist 跑：

### 阶段 1：链路可用（你已经完成）

- ✅ 三数据集统一输入格式
- ✅ Base / Hour / Δt embedding 能稳定训练

### 阶段 2：时间差权重（下一步）

- 产出：一个 “dt-attention baseline”
- 对比：Base vs dt-emb vs dt-attn
- 意义：证明“时间差作为权重”比“时间差作为 embedding”更合理

### 阶段 3：复现兴趣时钟 / LIC（核心）

- 产出：Clock 模块实现（最小可用版）
- 对比：Base、Hour、dt-emb、dt-attn、Clock、LIC
- 消融（任务书老师提到的）：去掉 GSU、去掉 ESU、去掉时间差权重、不同 bucket 数等

### 阶段 4：评估与论文表格

- 多数据集：MovieLens / Steam / Amazon
- 指标：HR@K、NDCG@K、MRR（按任务书）
- 产出：对比表 + 消融表 + 参数表

### 阶段 5：系统部署

- API：`/recommend?user_id=...`
- A/B：不同模型返回不同列表
- 可视化：兴趣时钟（你可以不画复杂图，做简单折线/热力图也行）

