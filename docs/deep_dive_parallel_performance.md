# 并行策略的性能建模与理论分析

本文档深入分析各种并行策略在PCIe架构下的性能特征，提供数学模型和优化理论。

---

## 1. 数据并行的扩展性理论

### 1.1 理想扩展模型

**基本假设**：
- 计算完全并行
- 通信完美overlap
- 无同步开销

**扩展效率公式**：

```
E(N) = S(N) / N = T(1) / (N × T(N))

其中：
- E(N)：N个设备的并行效率
- S(N)：加速比
- T(N)：使用N个设备的时间
```

**理想情况**：E(N) = 1 (完美线性扩展)

### 1.2 实际扩展模型（考虑通信）

**时间分解**：

```
T(N) = T_compute(N) + T_communicate(N) + T_sync(N)

其中：
T_compute(N) = T_compute(1) / (N × η_compute)
T_communicate(N) = f(N, algorithm, bandwidth)
T_sync(N) = N × t_barrier

η_compute：计算效率（通常0.85-0.95）
```

**Ring AllReduce通信时间**：

```
T_allreduce(N, S, B) = 2(N-1)/N × S/B + α(N-1)

其中：
- S：数据量（bytes）
- B：带宽（bytes/s）
- α：延迟（latency per hop）
- N：设备数

分析：
1. 当S很大时，带宽项主导：T ≈ 2S/B（与N无关！）
2. 当S很小时，延迟项主导：T ≈ α×N

临界点：S_critical = α×B×N/2
```

**实例计算（910C，8卡）**：

```python
# 参数
N = 8
B = 25e9  # 25 GB/s (实测PCIe带宽)
alpha = 2e-6  # 2 μs (实测hop延迟)

# 模型参数量 → 通信量
model_size_7B = 7e9 * 2  # 14 GB (FP16)

# 计算时间（假设batch=32, seq=2048）
flops_per_token = 2 * 7e9 * 12  # 前向+反向，估算
total_flops = flops_per_token * 32 * 2048
compute_time = total_flops / (320e12 * 8 * 0.85)  # 8卡，85%效率
# compute_time ≈ 2.3 秒

# 通信时间
comm_time = 2 * (N-1) / N * model_size_7B / B + alpha * (N-1)
# comm_time = 2 * 7/8 * 14e9 / 25e9 + 2e-6 * 7
#           = 0.98 + 0.000014
#           ≈ 0.98 秒

# 扩展效率
total_time = compute_time + comm_time  # 3.28 秒
ideal_time = compute_time / 8  # 0.29 秒（理想8卡）
efficiency = ideal_time / (total_time / 8)
#          = 0.29 / 0.41
#          ≈ 71%

# 结论：通信占比 = 0.98 / 3.28 ≈ 30%，严重影响扩展性
```

### 1.3 梯度累积的数学优化

**问题建模**：

给定：
- 计算时间（单micro-batch）：t_c
- 通信时间：t_m
- 梯度累积步数：G

```
总时间（无累积）：
T_0 = t_c + t_m

总时间（累积G步）：
T_G = G × t_c + t_m

加速比：
A = T_0 / T_G = (t_c + t_m) / (G × t_c + t_m)

求最优G（最大化吞吐量）：
吞吐量 = (G × batch_size) / T_G
       = (G × b) / (G×t_c + t_m)

对G求导：
d/dG[(G×b)/(G×t_c + t_m)] = b×t_m / (G×t_c + t_m)^2 > 0

因此：G越大越好！

但受限于：
1. 内存容量（梯度累积占内存）
2. 收敛性（等效大batch可能损害收敛）
3. 延迟（增加step时间）

实际最优G：
G_opt = min(
    sqrt(available_memory / gradient_size),
    critical_batch_size / micro_batch_size,
    max_acceptable_latency / t_c
)
```

**案例分析**：

```
场景：GPT-7B，8卡910C

参数：
- t_c = 2.3 / 8 = 0.29 秒（单卡单step）
- t_m = 0.98 秒
- 内存限制：G ≤ 16
- 收敛限制：effective_batch ≤ 2048，单卡batch=32，因此 G ≤ 8

候选值：G ∈ {1, 2, 4, 8}

计算吞吐量（samples/s）：
G=1: (1×32×8) / (0.29+0.98) = 201 samples/s
G=2: (2×32×8) / (0.58+0.98) = 328 samples/s  (+63%)
G=4: (4×32×8) / (1.16+0.98) = 477 samples/s  (+137%)
G=8: (8×32×8) / (2.32+0.98) = 780 samples/s  (+288%)

但：G=8时，effective_batch = 2048，可能影响收敛

权衡：G=4 是较好选择（477 samples/s，effective_batch=1024）
```

### 1.4 通信与计算Overlap分析

**理论模型**：

```
假设可以完美overlap，则：
T_overlap = max(T_compute, T_communicate)

实际overlap效率：
T_actual = T_compute + (1-η_overlap) × T_communicate

其中 η_overlap ∈ [0, 1]

PCIe架构的η_overlap限制：
1. DMA引擎独立 → 理论可以overlap
2. 但共享HBM带宽 → 实际竞争

测量数据（910C）：
- 纯计算：320 TFLOPS
- 纯通信：25 GB/s
- 同时进行：280 TFLOPS + 22 GB/s

带宽竞争：
HBM总带宽 = 1.2 TB/s
计算需求 = 280 TFLOPS / AI ≈ 1.0 TB/s（假设AI=280）
通信需求 = 22 GB/s（通过PCIe，但需要HBM读写）
实际HBM需求 = 1.0 + 0.022 ≈ 1.02 TB/s（略超限！）

因此：η_overlap ≈ 22/25 = 0.88（通信端）
```

**Bucket大小优化**：

```
问题：选择最优bucket大小以最大化overlap

小bucket：
- 优点：早开始通信，overlap多
- 缺点：通信次数多，overhead大

大bucket：
- 优点：通信次数少，效率高
- 缺点：必须等更多梯度ready，overlap少

模型：
N个参数，分为K个bucket，每个大小S=N/K

假设反向传播线性进行：
- 第i个bucket ready时间：t_i = i × T_backward / K
- 该bucket通信时间：t_comm = S / B = N / (K×B)

Timeline分析：
Bucket 1: 计算[0, T/K]，通信[T/K, T/K + N/(KB)]
Bucket 2: 计算[T/K, 2T/K]，通信[2T/K, 2T/K + N/(KB)]
...

Overlap条件：
t_comm < T_backward / K
即：N / (K×B) < T_backward / K
解得：K > N / (T_backward × B)

实例（910C，GPT-7B）：
N = 14 GB (FP16梯度)
T_backward = 1.5 秒
B = 25 GB/s

K_min = 14 / (1.5 × 25) = 0.37

因此 K ≥ 1 即可完全overlap！

但考虑overhead，实际最优：
K_opt ≈ 10-20（对应bucket大小 700MB-1.4GB）

DDP默认bucket_cap_mb = 25MB：
K = 14GB / 25MB = 560 个bucket（过多！）

推荐：bucket_cap_mb = 100-200 MB（针对大模型）
```

---

## 2. 模型并行的复杂度分析

### 2.1 张量并行的通信开销

**Megatron-LM式张量并行**：

单个Transformer层的通信pattern：

```
前向传播：
1. QKV投影（列并行）：
   输入：[b, s, h]，全局
   输出：[b, s, h/N]，分片
   通信：输入需要broadcast（或all-gather）
   量：b×s×h × dtype (如果需要)

2. Attention计算：
   本地计算，无通信

3. Attention输出（行并行）：
   输入：[b, s, h/N]，分片
   输出：[b, s, h]，全局
   通信：AllReduce
   量：b×s×h × dtype

4. FFN上投影（列并行）：
   通信：（已有全局输入）
   量：0

5. FFN下投影（行并行）：
   通信：AllReduce
   量：b×s×h × dtype

反向传播：对称，相同量

总通信量（单层，前向+反向）：
T_comm = 4 × b×s×h × dtype
```

**通信时间建模**：

```
对于TP=N（张量并行度为N）：

AllReduce时间：
t_ar = 2(N-1)/N × (b×s×h×dtype) / B + α(N-1)

单层总通信：
T_layer_comm = 4 × t_ar
             = 8(N-1)/N × (b×s×h×dtype) / B + 4α(N-1)

L层总通信：
T_total_comm = L × T_layer_comm

实例（BERT-Large，12层，TP=4）：
b=32, s=512, h=1024, dtype=2 bytes, L=12
B=25 GB/s, α=2μs

单层通信量 = 4 × 32×512×1024×2 = 128 MB
T_layer_comm = 8×3/4 × 128MB / 25GB/s + 4×2μs×3
             = 6 × 128MB / 25GB/s + 24μs
             = 30.7 ms + 0.024 ms
             ≈ 30.7 ms

12层总通信 = 12 × 30.7 = 368 ms

计算时间（假设）：
单层计算 = 2 × (b×s×h) × (4h×h) / (Peak_FLOPS/N)
         = 2 × 32×512×1024 × 4×1024×1024 / (320e12/4)
         ≈ 5.5 ms

12层总计算 = 66 ms

通信占比 = 368 / (368+66) = 85% ⚠️

结论：PCIe架构下，张量并行通信开销极大！
```

**张量并行 vs 数据并行对比**：

| 并行策略 | 通信频率 | 通信量/step | 通信占比 | 适用场景 |
|---------|---------|------------|---------|---------|
| 数据并行(DP) | 1次/step | Model_size | 30% | 首选 |
| 张量并行(TP=4) | 2L次/step | 4×b×s×h×L | 85% | 仅必要时 |
| 流水线并行(PP=4) | O(micro_batches) | b×s×h | 15% | 大模型推荐 |

### 2.2 流水线并行的Bubble分析

**GPipe调度**：

```
假设：
- P个pipeline stage
- M个micro-batch

Timeline（简化）：
Stage 0: [F0, F1, F2, ..., F_{M-1}][B0, B1, ..., B_{M-1}]
Stage 1:     [F0, F1, ..., F_{M-1}][B0, B1, ..., B_{M-1}]
...
Stage P-1:             ...[F0, F1, ..., F_{M-1}][B0, B1, ..., B_{M-1}]

Bubble（空闲时间）：
- 填充阶段：(P-1)个时间slot
- 排空阶段：(P-1)个时间slot
- 总bubble：2(P-1)个slot

有效计算：M个slot

Bubble比例：
Bubble_ratio = 2(P-1) / (M + 2(P-1))

减少bubble：增大M

实例：
P=4, M=8: Bubble = 2×3/(8+6) = 43%  ⚠️ 太高
P=4, M=32: Bubble = 6/(32+6) = 16%  ✓ 可接受
P=4, M=128: Bubble = 6/(128+6) = 4% ✓ 很好

但M过大会增加内存占用（需保存M份中间激活）
```

**1F1B（One Forward One Backward）调度**：

```
改进：前向和反向交织，减少bubble

Timeline：
Stage 0: [F0, F1, ..., F_{P-1}][F_P, B0][F_{P+1}, B1]...[B_{M-1}]
Stage 1:     [F0, F1, ..., F_{P-2}][F_{P-1}, B0][F_P, B1]...
...

Bubble分析：
- Warmup：(P-1)个slot（只有前向）
- Steady state：前向+反向overlap，无bubble
- Cooldown：(P-1)个slot（只有反向）

总bubble：2(P-1)个slot（与GPipe相同）
但内存占用：只需保存P份激活（vs GPipe的M份）

因此：1F1B严格优于GPipe！
```

**通信开销**：

```
流水线并行的通信：
- 相邻stage间传递激活值
- 数据量：b×s×h×dtype（每个micro-batch）

单个micro-batch通信时间：
t_comm = (b×s×h×dtype) / B + α

实例（b=4, s=512, h=1024, dtype=2, B=25GB/s）：
数据量 = 4×512×1024×2 = 4 MB
t_comm = 4MB / 25GB/s + 2μs ≈ 160μs + 2μs = 162μs

计算时间（单micro-batch，单stage）：
假设stage计算约10ms

通信占比 = 0.162 / 10 = 1.6%  ✓ 很低！

结论：流水线并行的通信开销远小于张量并行
```

### 2.3 3D并行的最优配置

**目标**：给定N个GPU，选择DP、TP、PP的最优组合。

**约束**：
```
DP × TP × PP = N
```

**优化目标**：
```
最小化：T_total = T_compute + T_communicate

T_compute = T_single_gpu / (DP × TP × PP × η_compute)
          ≈ T_single_gpu / (N × 0.85)  (相对固定)

T_communicate = T_DP + T_TP + T_PP

其中：
T_DP = 2(DP-1)/DP × Model_size / B + α(DP-1)
T_TP = 8(TP-1)/TP × L×b×s×h×dtype / B + 4L×α(TP-1)
T_PP ≈ 2(PP-1) × (b_micro×s×h×dtype / B + α)
```

**启发式规则**（基于通信开销最小化）：

```
1. 优先使用DP（通信最少）
   DP = max(N / (TP×PP), 1)

2. 仅在内存不足时使用TP
   TP = min(2 或 4, 必要时)
   原因：TP通信开销大，但可减少每GPU内存

3. PP用于超大模型
   PP = N / (DP×TP)

示例配置：

64 GPUs, GPT-175B:
- 单GPU放不下 → 必须MP
- 模型大小：175B × 2 bytes = 350 GB
- 单GPU内存：32 GB

需要MP度：350 / 32 ≈ 11
选择：TP=2, PP=8 → 16-way MP
则：DP = 64 / 16 = 4

配置：DP=4, TP=2, PP=8

验证内存：
- 每GPU参数：175B / (2×8) = 10.9B → 21.8GB ✓
- 加上激活、优化器状态等 ≈ 30GB ✓

通信开销估算：
T_DP ≈ 700ms (allreduce 350GB)
T_TP ≈ 2000ms (频繁小通信)
T_PP ≈ 200ms (pipeline通信)
总通信 ≈ 2.9s

计算时间 ≈ 8s (假设)
总时间 ≈ 11s
效率 ≈ 8 / 11 = 73%
```

**权衡分析**：

| 策略 | 内存节省 | 通信开销 | 实现复杂度 | PCIe友好度 |
|-----|---------|---------|----------|-----------|
| DP | 低 | 最低 | 简单 | ★★★★★ |
| TP | 高 | 最高 | 中等 | ★☆☆☆☆ |
| PP | 高 | 中等 | 复杂 | ★★★☆☆ |
| ZeRO | 高 | 中等 | 简单 | ★★★★☆ |

**910C推荐**：
1. 优先DP + ZeRO（内存不足时）
2. 必要时DP + PP（避免TP）
3. 极端情况DP + TP + PP（TP度≤4）

---

## 3. ZeRO的理论分析

### 3.1 内存分解

**训练内存占用**：

```
总内存 = Model_States + Activations + Temp_Buffers

Model_States = Parameters + Gradients + Optimizer_States

对于AdamW优化器：
Parameters: Ψ
Gradients: Ψ
Optimizer_States: 2Ψ (momentum + variance)
总计：4Ψ

示例：GPT-7B，FP16
Ψ = 7B × 2 bytes = 14 GB
Model_States = 4 × 14 GB = 56 GB  (超出单卡32GB！)

Activations (取决于batch size和序列长度)：
对于batch=32, seq=2048, hidden=4096:
每层激活 ≈ 32 × 2048 × 4096 × 2 bytes × 4 (QKVO) = 2 GB
32层 = 64 GB  (需要gradient checkpointing)

使用checkpointing后 ≈ 8 GB
```

### 3.2 ZeRO Stage 1-3分析

**Stage 1（Optimizer State Partitioning）**：

```
优化器状态分片到N个GPU

每GPU内存：
Ψ (parameters) + Ψ (gradients) + 2Ψ/N (optimizer states)
= (2 + 2/N)Ψ

N=8: 内存 = 2.25Ψ (节省 43.75%)

通信开销：
- 需要all-gather更新后的参数
- 量：Ψ
- 时间：≈ 2(N-1)/N × Ψ/B (一次AllGather)
- 与标准DP的AllReduce相同！

结论：Stage 1几乎无额外通信开销
```

**Stage 2（Gradient Partitioning）**：

```
优化器状态 + 梯度分片

每GPU内存：
Ψ + Ψ/N + 2Ψ/N = (1 + 3/N)Ψ

N=8: 内存 = 1.375Ψ (节省 65.6%)

通信开销：
- 反向传播时：reduce-scatter梯度
- 优化器步骤后：all-gather参数

reduce-scatter量：Ψ
all-gather量：Ψ
总量：2Ψ

对比标准DP的allreduce：2(N-1)/N × Ψ ≈ 2Ψ (N大时)

结论：Stage 2通信量与DP相近
```

**Stage 3（Parameter Partitioning）**：

```
所有model states分片

每GPU内存：
Ψ/N + Ψ/N + 2Ψ/N = 4Ψ/N

N=8: 内存 = 0.5Ψ (节省 87.5%)

通信开销：
- 前向：all-gather参数 (每层)
- 反向：all-gather参数 + reduce-scatter梯度 (每层)
- L层总计：all-gather量 = 2L×Ψ/L = 2Ψ
            reduce-scatter量 = Ψ
            总量 = 3Ψ

对比DP：增加50%通信量 ⚠️

对于PCIe架构：
假设Ψ=14GB, B=25GB/s, N=8:
额外通信时间 = 0.5 × 14GB / 25GB/s = 280ms

如果计算时间 = 2s：
通信增加 = 280/2000 = 14%

可接受，但Stage 2更优
```

**Stage选择建议（910C）**：

| 模型大小 | 单卡可放下？ | 推荐Stage | 原因 |
|---------|------------|----------|------|
| <10B | 是 | Stage 0/1 | 无需分片 |
| 10B-20B | 临界 | Stage 1/2 | 平衡内存和通信 |
| 20B-50B | 否 | Stage 2 | 最佳权衡 |
| >50B | 远超 | Stage 3 + PP | 必要之恶 |

### 3.3 ZeRO-Offload分析

**CPU Offload策略**：

```
目标：将优化器状态offload到CPU内存

内存节省：2Ψ (优化器状态)

额外通信：
- 梯度：GPU → CPU (Ψ)
- 参数更新：CPU → GPU (Ψ)

PCIe带宽限制：
B_pcie ≈ 32 GB/s (理论)
实测 ≈ 25 GB/s

时间成本：
T_offload = 2Ψ / B_pcie

实例：GPT-7B (Ψ=14GB)
T_offload = 2 × 14GB / 25GB/s = 1.12s

如果计算时间 = 2s：
开销 = 1.12 / 2 = 56%  ⚠️ 很大！

结论：
- CPU offload在PCIe架构下开销大
- 仅在内存极度受限时使用
- 优先考虑ZeRO Stage 2 + Gradient Checkpointing
```

---

## 4. 通信拓扑优化

### 4.1 拓扑感知的Rank分配

**问题**：如何分配rank以最小化通信延迟？

**拓扑图**：

```
910C服务器（8卡）：
CPU Socket 0 → PCIe Switch 0 → NPU 0, 1, 2, 3
CPU Socket 1 → PCIe Switch 1 → NPU 4, 5, 6, 7

通信延迟矩阵（相对值）：
     0   1   2   3   4   5   6   7
0  [ 0   1   1   1   3   3   3   3 ]
1  [ 1   0   1   1   3   3   3   3 ]
2  [ 1   1   0   1   3   3   3   3 ]
3  [ 1   1   1   0   3   3   3   3 ]
4  [ 3   3   3   3   0   1   1   1 ]
5  [ 3   3   3   3   1   0   1   1 ]
6  [ 3   3   3   3   1   1   0   1 ]
7  [ 3   3   3   3   1   1   1   0 ]
```

**DP rank分配**：

```
目标：最小化AllReduce通信时间

Ring AllReduce的通信pattern：
Ring: 0→1→2→...→7→0

总延迟 = Σ(i=0 to N-1) latency[i, (i+1)%N]

坏的分配（默认顺序0-7）：
延迟 = 1+1+1+3+1+1+1+3 = 12

好的分配（拓扑感知）：
Ring: 0→1→2→3→4→5→6→7→0
改为：0→1→2→3→7→6→5→4→0
延迟 = 1+1+1+3+1+1+1+3 = 12（相同）

最优分配（分组）：
Group0: 0→1→2→3→0 (延迟=4)
Group1: 4→5→6→7→4 (延迟=4)
使用Hierarchical AllReduce：
Step1: 组内AllReduce
Step2: 跨组reduce (0-4, 1-5, 2-6, 3-7)
Step3: 组内broadcast

总延迟 = 4 + 4×3 + 4 = 20  ⚠️ 更差！

结论：对于8卡，标准Ring已经很优
但对于64卡（8机×8卡），分层算法会更优
```

### 4.2 多机通信优化

**64卡配置（8机×8卡）**：

```
网络拓扑：
- 机内：PCIe（25 GB/s）
- 机间：IB/RoCE（100 Gb/s = 12.5 GB/s实际约10 GB/s）

延迟：
- 机内：2 μs
- 机间：20 μs (10倍)

优化策略：

1. 分层AllReduce
   Step1：机内AllReduce（8卡Ring）
   Step2：机间AllReduce（8机Ring，每机1个代表）
   Step3：机内Broadcast

   通信量分析（数据量S）：
   Step1: 每机内 2(8-1)/8 × S/8 = 7S/32 (并行)
   Step2: 机间 2(8-1)/8 × S = 7S/4
   Step3: 每机内 S/8 (broadcast)

   时间：
   T1 = 7S/32 / 25GB/s = 0.00875 S
   T2 = 7S/4 / 10GB/s = 0.175 S
   T3 = S/8 / 25GB/s = 0.005 S
   总计 ≈ 0.189 S/GB

   对比单层Ring（64卡）：
   T_ring = 2(64-1)/64 × S / 10GB/s = 0.197 S/GB

   提升：(0.197-0.189)/0.197 ≈ 4%

2. 二维Torus拓扑（8×8网格）
   每个节点只需与4个邻居通信
   通信步数：O(√N) vs O(N)
   但需要定制AllReduce算法

实际建议：
- 使用HCCL的拓扑感知自动优化
- 设置HCCL_TOPO_CONFIG环境变量
- 对于>16卡，分层AllReduce通常更优
```

---

## 5. 性能预测模型

### 5.1 端到端训练时间预测

**输入参数**：
- 模型：参数量Ψ，层数L，hidden size h
- 硬件：N个GPU，计算峰值P，带宽B
- 配置：batch size b，序列长度s，并行策略(DP, TP, PP)

**模型**：

```python
def predict_training_time(model, hardware, config):
    """预测单step训练时间"""

    # 1. 计算FLOPs
    # Transformer: FLOPs ≈ 96 b s h^2 L (近似)
    flops_forward = 96 * b * s * h**2 * L
    flops_backward = 2 * flops_forward  # 反向约2×前向
    total_flops = flops_forward + flops_backward

    # 2. 计算时间
    compute_time = total_flops / (N * P * efficiency)

    # 3. 通信时间
    if config.strategy == "DP":
        comm_time = 2 * (N-1) / N * (model_bytes / B)
    elif config.strategy == "TP":
        comm_time = 8 * (TP-1) / TP * L * (b*s*h*dtype / B)
    elif config.strategy == "PP":
        comm_time = 2 * (PP-1) * (b_micro*s*h*dtype / B)

    # 4. 梯度累积影响
    effective_compute = compute_time * gradient_accum_steps
    effective_comm = comm_time  # 只需一次通信

    return effective_compute + effective_comm

# 示例
model = GPT(params=7e9, layers=32, hidden=4096)
hardware = Ascend910C(num=8, peak_flops=320e12, bandwidth=25e9)
config = Config(batch=32, seqlen=2048, strategy="DP", grad_accum=4)

time_per_step = predict_training_time(model, hardware, config)
# 预测：2.8秒/step
```

### 5.2 超参数调优指导

**吞吐量优化**：

```
目标：最大化 samples/second

Throughput = (batch_size × grad_accum × N) / T_step

受限于：
1. 内存：batch_size × seqlen × hidden × layers ≤ Memory
2. 收敛性：effective_batch = batch×accum×N ≤ Critical_batch
3. 数值稳定性：learning_rate ∝ √effective_batch

优化算法：
1. 二分搜索最大batch_size（内存约束）
2. 计算最大grad_accum（收敛约束）
3. 预测吞吐量
4. 调整batch和accum的组合以最大化吞吐量

伪代码：
```python
def optimize_hyperparams(model, hardware, constraints):
    max_batch = binary_search_max_batch(model, hardware)

    best_throughput = 0
    best_config = None

    for batch in range(1, max_batch + 1):
        max_accum = constraints.critical_batch // (batch * hardware.num_gpus)

        for accum in range(1, max_accum + 1):
            time = predict_training_time(model, hardware,
                                        Config(batch, accum))
            throughput = (batch * accum * hardware.num_gpus) / time

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = (batch, accum)

    return best_config

# 输出：batch=32, grad_accum=4 → 780 samples/s
```

---

## 6. 总结

### 6.1 关键数学洞察

1. **扩展效率受限于通信比例**
   ```
   E(N) ≤ T_compute / (T_compute + T_communicate)
   ```
   PCIe架构：T_comm/T_comp ≈ 0.3-0.5 → E(8) ≈ 67-77%

2. **梯度累积的收益递减**
   ```
   Speedup(G) = (1 + r) / (G + r)，其中 r = T_comm/T_comp
   ```
   G增大时，speedup趋于1/r的上限

3. **Bubble时间与micro-batch数量**
   ```
   Bubble% = 2(P-1) / (M + 2(P-1))
   ```
   M需 >> 2P才能有效降低bubble

4. **ZeRO内存-通信权衡**
   ```
   内存：4Ψ/N^{stage}
   通信：(1 + stage/2) × 2Ψ
   ```

### 6.2 910C优化决策树

```
模型能否单卡放下？
├─ 是 → 使用DP + Grad Accum(4-8)
│        └─ 内存紧张？→ ZeRO Stage 1
│
└─ 否 → 参数量多大？
         ├─ <30B → DP + ZeRO Stage 2 + Checkpointing
         ├─ 30-70B → DP + PP(2-4) + ZeRO Stage 2
         └─ >70B → DP + PP(4-8) + TP(2) + ZeRO Stage 3
                   └─ 仍不够？→ Offload + 减小batch
```

这些数学模型和分析为实际配置选择提供了定量依据。
