# 昇腾910C并行调优指南

## 核心挑战：PCIe vs NVLink

昇腾910C使用PCIe 4.0互连，与NVIDIA GPU的NVLink相比存在显著的带宽差异，这要求我们采用不同的并行策略。

### 带宽对比

| 互连类型 | 单卡带宽 | 8卡总带宽 | 延迟 |
|---------|---------|----------|-----|
| PCIe 4.0 x16 | 32 GB/s | ~256 GB/s | 1-2 μs |
| NVLink 3.0 (A100) | 600 GB/s | ~4800 GB/s | 0.5 μs |
| **带宽比** | **1:18** | **1:18** | **2:1** |

**关键启示**：PCIe带宽约为NVLink的1/18，因此通信密集型并行策略需要重新设计。

---

## 1. 数据并行 (Data Parallelism, DP)

### 1.1 为什么数据并行是910C的首选？

**优势**：
- ✅ 通信频率低：每个step只需一次AllReduce
- ✅ 通信量可预测：与模型参数量成正比
- ✅ 扩展性好：理论上可扩展到任意卡数
- ✅ 实现简单：框架原生支持

**PCIe架构下的适用场景**：
- CV模型训练（ResNet, EfficientNet等）
- 中小规模语言模型（BERT, GPT-2等）
- batch size可以增大的任务

### 1.2 数据并行优化技巧

#### 1.2.1 梯度累积（Gradient Accumulation）

**核心思路**：通过累积多个micro-batch的梯度，减少通信频率。

```python
# PyTorch + Torch-NPU 示例
import torch
import torch_npu

# 配置
gradient_accumulation_steps = 4  # PCIe场景建议4-8
micro_batch_size = 32
effective_batch_size = micro_batch_size * gradient_accumulation_steps * world_size

optimizer.zero_grad()
for step in range(gradient_accumulation_steps):
    # 前向传播
    outputs = model(inputs[step])
    loss = criterion(outputs, labels[step])

    # 梯度缩放
    loss = loss / gradient_accumulation_steps

    # 反向传播（不同步）
    loss.backward()

# 所有micro-batch完成后，进行一次AllReduce
optimizer.step()
optimizer.zero_grad()
```

**效果分析**：
- 通信次数：减少4-8倍
- 通信时间占比：从30% -> 5-10%
- 内存占用：略有增加（梯度累积）

#### 1.2.2 通信与计算重叠（Overlap）

HCCL支持通信与反向计算重叠：

```python
import torch.distributed as dist

# 启用梯度bucketing和overlap
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,  # 调整bucket大小
    gradient_as_bucket_view=True,  # 减少内存拷贝
    broadcast_buffers=False,  # 不同步buffer
    find_unused_parameters=False  # 如果确定所有参数都用到，关闭以提升性能
)
```

**Bucket大小调优**：
- **小bucket (10-25 MB)**：更早开始通信，overlap效果好
- **大bucket (50-100 MB)**：通信次数少，适合高延迟网络
- **910C建议**：25-50 MB（平衡通信次数和overlap）

#### 1.2.3 混合精度梯度通信

FP16梯度可以减少50%的通信量：

```python
from apex import amp  # 或使用torch.cuda.amp

# 使用FP16进行AllReduce
model, optimizer = amp.initialize(
    model,
    optimizer,
    opt_level='O2',  # 几乎全FP16
    loss_scale='dynamic',
    keep_batchnorm_fp32=True
)
```

**通信量对比**：
- FP32梯度：4 bytes/参数
- FP16梯度：2 bytes/参数
- **节省50%带宽**

### 1.3 HCCL配置优化

#### 1.3.1 AllReduce算法选择

HCCL支持多种AllReduce算法：

```bash
# 环境变量配置
export HCCL_ALGO=ring              # Ring算法（默认）
export HCCL_ALGO=tree              # Tree算法
export HCCL_ALGO=ring,tree         # 自动选择

# Ring算法特点
# - 带宽利用率高
# - 适合大消息
# - 延迟随卡数线性增长

# Tree算法特点
# - 延迟低
# - 适合小消息
# - 带宽利用率相对低
```

**910C建议**：
- 默认使用Ring算法
- 小模型(<100M参数)可尝试Tree
- 大模型(>1B参数)必用Ring

#### 1.3.2 拓扑感知

```bash
# 指定NPU拓扑
export HCCL_SOCKET_IFNAME=eth0     # 指定网络接口
export HCCL_WHITELIST_DISABLE=1    # 禁用白名单检查

# 打印拓扑信息
export HCCL_DETERMINISTIC=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

#### 1.3.3 性能调优

```bash
# 通信压缩
export HCCL_GRAD_COMPRESSION=1     # 梯度压缩（FP16）

# 通信并发
export HCCL_STREAM_NUM=2           # 增加通信流数量

# 内存优化
export HCCL_BUFFSIZE=512           # 通信buffer大小(MB)
```

---

## 2. 模型并行 (Model Parallelism, MP)

### 2.1 PCIe架构下的挑战

**问题**：
- ❌ 激活值传输开销大
- ❌ 前向/反向传播需要频繁同步
- ❌ Pipeline bubble严重（流水线并行）

**适用场景**：
- 模型无法放入单卡（>32GB）
- 必须配合其他并行策略使用

### 2.2 张量并行（Tensor Parallelism）

**原理**：将单个Transformer层的参数切分到多卡。

```python
# Megatron-LM风格的张量并行
# 列并行Linear
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.out_features_per_rank = out_features // world_size

        # 每个rank只保存部分权重
        self.weight = torch.nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )

    def forward(self, x):
        # 本地矩阵乘法
        output = torch.matmul(x, self.weight.t())
        # 无需通信（下一层AllReduce）
        return output

# 行并行Linear
class RowParallelLinear(torch.nn.Module):
    def forward(self, x):
        output = torch.matmul(x, self.weight.t())
        # AllReduce合并结果
        dist.all_reduce(output)
        return output
```

**PCIe场景优化**：
- 仅在必要时使用（模型>32GB）
- 张量并行度不超过4（通信开销控制）
- 优先使用流水线并行

### 2.3 流水线并行（Pipeline Parallelism）

**推荐方案**：GPipe / 1F1B调度

```python
# 使用MindSpore的流水线并行
from mindspore import context
from mindspore.nn import PipelineCell

# 划分层到不同设备
context.set_auto_parallel_context(
    parallel_mode="semi_auto_parallel",
    pipeline_stages=4  # 4级流水线
)

# 定义micro_batch数量（减少bubble）
micro_batch_num = 16  # 增加micro_batch减少bubble

# 模型定义
class PipelineModel(nn.Cell):
    def __init__(self):
        super().__init__()
        # Stage 0
        self.layer1 = TransformerLayer().to_device(0)
        # Stage 1
        self.layer2 = TransformerLayer().to_device(1)
        # ...
```

**Bubble优化**：
- Micro-batch数量 = 流水线阶段数 × 4
- 使用1F1B调度而非GPipe
- 激活重计算节省内存

**PCIe下的流水线建议**：
- Pipeline stages: 2-4（不要太多）
- Micro-batches: 8-16
- 激活checkpointing: 必须启用

---

## 3. 混合并行策略

### 3.1 3D并行（DP + PP + TP）

针对超大模型（>10B参数）的并行策略：

```
总卡数：64 (8机×8卡)
模型：GPT-175B

并行配置：
- Pipeline Parallel (PP) = 8   # 8个流水线阶段
- Tensor Parallel (TP) = 2     # 每个阶段内2路张量并行
- Data Parallel (DP) = 4       # 4路数据并行

拓扑：
DP组0: [PP0-TP0, PP0-TP1] [PP1-TP0, PP1-TP1] ... [PP7-TP0, PP7-TP1]
DP组1: [PP0-TP0, PP0-TP1] [PP1-TP0, PP1-TP1] ... [PP7-TP0, PP7-TP1]
...
```

**910C配置建议**：

| 模型规模 | DP | PP | TP | Micro-batch | 备注 |
|---------|----|----|----|-----------|----|
| 1B | 8 | 1 | 1 | 4 | 纯数据并行 |
| 7B | 8 | 1 | 1 | 4 | 纯数据并行 |
| 13B | 4 | 2 | 1 | 8 | DP+PP |
| 30B | 4 | 2 | 2 | 8 | DP+PP+TP |
| 70B | 2 | 4 | 2 | 16 | DP+PP+TP |
| 175B | 2 | 8 | 2 | 16 | DP+PP+TP |

### 3.2 ZeRO数据并行

DeepSpeed ZeRO可以在数据并行基础上节省内存：

```python
# DeepSpeed配置
ds_config = {
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 2,  # PCIe场景建议Stage 2
        # Stage 1: 优化器状态分片
        # Stage 2: 优化器+梯度分片
        # Stage 3: 优化器+梯度+参数分片（通信量大，不推荐）

        "contiguous_gradients": True,
        "overlap_comm": True,  # 通信计算重叠
        "reduce_bucket_size": 50000000,
        "allgather_bucket_size": 50000000,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
    }
}
```

**ZeRO Stage选择（PCIe场景）**：

| Stage | 内存节省 | 通信开销 | 910C建议 |
|-------|---------|---------|---------|
| Stage 1 | 4x | 低 | ✅ 推荐 |
| Stage 2 | 8x | 中 | ✅ 可用 |
| Stage 3 | 线性 | 高 | ⚠️ 谨慎使用 |

**原因**：Stage 3会分片参数，需要频繁AllGather，PCIe带宽不足。

---

## 4. 通信优化技巧

### 4.1 梯度压缩

```python
# PowerSGD梯度压缩
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

# 注册压缩hook
model = DDP(model, ...)
state = powerSGD_hook.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=4,  # 压缩秩
    start_powerSGD_iter=10,  # 前10个iter不压缩
)
model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)
```

**压缩效果**：
- 通信量减少：5-10x
- 精度损失：<0.5%
- 适合：大模型训练

### 4.2 局部SGD（Local SGD）

```python
# 每N步才进行一次梯度同步
local_steps = 4  # PCIe场景可设置4-8

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        loss = model(batch)
        loss.backward()

        # 仅在指定步数才同步
        if (step + 1) % local_steps == 0:
            # 同步梯度
            for param in model.parameters():
                dist.all_reduce(param.grad)
                param.grad /= world_size

        optimizer.step()
        optimizer.zero_grad()
```

**效果**：
- 通信次数：减少4-8倍
- 收敛性：可能略有下降
- 适合：对收敛不敏感的任务

### 4.3 通信融合

```python
# HCCL通信融合
export HCCL_FUSION=1
export HCCL_FUSION_THRESHOLD_MB=64  # 融合阈值

# 或在代码中手动融合
def fused_allreduce(tensor_list):
    # 将多个小tensor拼接
    flat_tensor = torch.cat([t.flatten() for t in tensor_list])

    # 一次AllReduce
    dist.all_reduce(flat_tensor)

    # 拆分回去
    offset = 0
    for t in tensor_list:
        numel = t.numel()
        t.copy_(flat_tensor[offset:offset+numel].view_as(t))
        offset += numel
```

---

## 5. 单机多卡配置示例

### 5.1 PyTorch + Torch-NPU

```python
import torch
import torch.distributed as dist
import torch_npu

def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend='hccl',  # 使用HCCL后端
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # 设置设备
    torch_npu.npu.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # 模型定义
    model = YourModel().to(f'npu:{rank}')
    model = DDP(model, device_ids=[rank])

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

    cleanup()

if __name__ == '__main__':
    world_size = 8
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

### 5.2 MindSpore

```python
from mindspore import context
from mindspore.communication import init

# 设置并行模式
context.set_auto_parallel_context(
    parallel_mode="data_parallel",
    gradients_mean=True,
    device_num=8,
)

# 初始化HCCL
init()

# 模型定义
model = YourModel()

# 自动并行
model = Model(
    network=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics={'acc'}
)

# 训练
model.train(epoch=10, train_dataset=dataset)
```

### 5.3 启动脚本

```bash
#!/bin/bash

# 8卡训练启动脚本
export RANK_SIZE=8
export RANK_TABLE_FILE=/path/to/rank_table.json

# HCCL配置
export HCCL_ALGO=ring
export HCCL_GRAD_COMPRESSION=1
export HCCL_STREAM_NUM=2

# 性能优化
export COMBINED_ENABLE=1  # 算子融合
export TASK_QUEUE_ENABLE=1  # 任务队列

# 启动训练
for((i=0;i<$RANK_SIZE;i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$i

    python train.py \
        --distributed \
        --world_size=$RANK_SIZE \
        --rank=$RANK_ID \
        --batch_size=32 \
        --gradient_accumulation_steps=4 \
        > logs/train_rank${i}.log 2>&1 &
done

wait
```

---

## 6. 多机多卡配置

### 6.1 Rank Table配置

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "192.168.1.100",
            "device": [
                {"device_id": "0", "device_ip": "192.168.100.100", "rank_id": "0"},
                {"device_id": "1", "device_ip": "192.168.100.101", "rank_id": "1"},
                {"device_id": "2", "device_ip": "192.168.100.102", "rank_id": "2"},
                {"device_id": "3", "device_ip": "192.168.100.103", "rank_id": "3"},
                {"device_id": "4", "device_ip": "192.168.100.104", "rank_id": "4"},
                {"device_id": "5", "device_ip": "192.168.100.105", "rank_id": "5"},
                {"device_id": "6", "device_ip": "192.168.100.106", "rank_id": "6"},
                {"device_id": "7", "device_ip": "192.168.100.107", "rank_id": "7"}
            ]
        },
        {
            "server_id": "192.168.1.101",
            "device": [
                {"device_id": "0", "device_ip": "192.168.101.100", "rank_id": "8"},
                {"device_id": "1", "device_ip": "192.168.101.101", "rank_id": "9"},
                {"device_id": "2", "device_ip": "192.168.101.102", "rank_id": "10"},
                {"device_id": "3", "device_ip": "192.168.101.103", "rank_id": "11"},
                {"device_id": "4", "device_ip": "192.168.101.104", "rank_id": "12"},
                {"device_id": "5", "device_ip": "192.168.101.105", "rank_id": "13"},
                {"device_id": "6", "device_ip": "192.168.101.106", "rank_id": "14"},
                {"device_id": "7", "device_ip": "192.168.101.107", "rank_id": "15"}
            ]
        }
    ],
    "status": "completed"
}
```

### 6.2 多机启动

```bash
# 节点0
bash start_train.sh 0 2

# 节点1
bash start_train.sh 1 2
```

---

## 7. 性能分析与调优

### 7.1 通信时间分析

```python
import time
import torch.distributed as dist

# 测量AllReduce时间
def benchmark_allreduce(size_mb, iterations=100):
    tensor_size = size_mb * 1024 * 1024 // 4  # FP32
    tensor = torch.randn(tensor_size).npu()

    # 预热
    for _ in range(10):
        dist.all_reduce(tensor)

    # 测量
    torch_npu.npu.synchronize()
    start = time.time()

    for _ in range(iterations):
        dist.all_reduce(tensor)

    torch_npu.npu.synchronize()
    elapsed = time.time() - start

    bandwidth = (size_mb * iterations) / elapsed
    print(f"Size: {size_mb}MB, Bandwidth: {bandwidth:.2f} MB/s")

# 测试不同消息大小
for size in [1, 10, 100, 500, 1000]:
    benchmark_allreduce(size)
```

### 7.2 Profiling

```bash
# 使用msprof profiling
msprof --application="python train.py" \
       --output=./profiling_result \
       --ai-core=on \
       --task-trace=on \
       --hccl-trace=on  # 通信trace

# 分析结果
# 查看 profiling_result/hccl_trace.json
```

**关键指标**：
- **通信时间占比**：应 <20%
- **计算时间占比**：应 >70%
- **Bubble时间**：流水线并行中应 <10%

---

## 8. 常见问题与解决方案

### 问题1：通信时间过长

**症状**：通信占比 >30%

**解决方案**：
1. 增加梯度累积步数
2. 启用梯度压缩
3. 检查网络拓扑，避免跨交换机通信
4. 减少模型并行度，增加数据并行度

### 问题2：显存不足

**症状**：OOM错误

**解决方案**：
1. 启用ZeRO Stage 2
2. 激活重计算（Gradient Checkpointing）
3. 减小batch size，增加梯度累积
4. 使用CPU offload（慎用，PCIe带宽有限）

### 问题3：扩展性差

**症状**：8卡速度 < 4卡速度 × 2

**解决方案**：
1. 检查是否有全局同步点（如指标计算）
2. 优化数据加载（增加num_workers）
3. 检查HCCL配置
4. Profile找到瓶颈

---

## 9. 最佳实践总结

### 9.1 推荐配置（8卡训练）

```python
# 训练配置
config = {
    # 并行策略
    "data_parallel": True,
    "model_parallel": False,  # 仅大模型启用

    # Batch配置
    "micro_batch_size": 32,
    "gradient_accumulation_steps": 4,  # 有效batch=32*4*8=1024

    # 混合精度
    "fp16": True,
    "bf16": False,  # 或选择BF16

    # 通信优化
    "gradient_compression": True,
    "overlap_comm": True,
    "bucket_size_mb": 25,

    # HCCL配置
    "hccl_algo": "ring",
    "hccl_stream_num": 2,
}
```

### 9.2 优化检查清单

- [ ] 混合精度已启用（FP16/BF16）
- [ ] 梯度累积步数 >= 4
- [ ] DDP已配置通信overlap
- [ ] HCCL算法选择正确
- [ ] 数据加载不是瓶颈
- [ ] 通信时间占比 < 20%
- [ ] AI Core利用率 > 80%
- [ ] 显存使用率 > 80%

### 9.3 针对不同场景的建议

| 场景 | 推荐策略 | 关键参数 |
|-----|---------|---------|
| CV训练 (ResNet) | 纯DP | grad_accum=4, fp16=True |
| NLP训练 (BERT) | DP | grad_accum=8, bf16=True |
| 大模型 (<13B) | DP + PP | pp_degree=2, grad_accum=8 |
| 大模型 (13B-70B) | DP + PP + TP | pp=4, tp=2, grad_accum=16 |
| 推理服务 | 单卡 | batch_size=max, fp16/int8 |

---

## 10. 总结

昇腾910C的PCIe架构要求我们：

1. **优先数据并行**：DP是PCIe架构下最友好的策略
2. **减少通信频率**：梯度累积、局部SGD
3. **压缩通信量**：FP16梯度、PowerSGD
4. **重叠通信计算**：DDP bucket、HCCL overlap
5. **谨慎模型并行**：仅大模型使用，且限制并行度

**核心原则**：用计算换通信，用内存换通信。

通过合理的并行策略配置，910C在大多数训练任务上可以达到与NVLink GPU相近的扩展效率。
