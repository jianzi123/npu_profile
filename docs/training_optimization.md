# 昇腾910C训练优化指南

本文档涵盖在昇腾910C上进行深度学习训练的各种优化技术，从混合精度到内存优化，从算子融合到数据预处理。

---

## 1. 混合精度训练

### 1.1 为什么必须使用混合精度？

昇腾910C的性能特点：
- FP16/BF16: ~320 TFLOPS
- FP32: ~20 TFLOPS
- **性能差异：16倍**

**结论**：不使用混合精度几乎是在浪费910C的算力。

### 1.2 FP16 vs BF16

| 特性 | FP16 | BF16 |
|-----|------|------|
| 指数位 | 5位 | 8位（同FP32） |
| 尾数位 | 10位 | 7位 |
| 动态范围 | ±65504 | ±3.4e38 |
| 精度 | 高 | 中 |
| 数值稳定性 | 需要loss scaling | 天然稳定 |
| 适用场景 | CV模型 | 大语言模型 |

**选择建议**：
- **CV任务（ResNet, EfficientNet等）**：FP16（精度高）
- **NLP任务（BERT, GPT等）**：BF16（范围大，稳定）
- **大模型训练**：BF16（避免溢出）

### 1.3 PyTorch混合精度实现

#### 1.3.1 使用torch.cuda.amp（推荐）

```python
import torch
import torch_npu
from torch.cuda.amp import autocast, GradScaler

# 创建模型和优化器
model = YourModel().npu()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 创建GradScaler（FP16需要，BF16不需要）
scaler = GradScaler()

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()

    # 自动混合精度上下文
    with autocast(dtype=torch.float16):  # 或torch.bfloat16
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    # 梯度缩放和反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 1.3.2 使用APEX（更多控制）

```python
from apex import amp

model = YourModel().npu()
optimizer = torch.optim.AdamW(model.parameters())

# 初始化amp
model, optimizer = amp.initialize(
    model,
    optimizer,
    opt_level='O2',  # O0:FP32, O1:混合, O2:几乎全FP16, O3:全FP16
    loss_scale='dynamic',  # 动态loss scaling
    keep_batchnorm_fp32=True,  # BN保持FP32
    master_weights=True  # 保留FP32权重副本
)

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(inputs)

    # 使用amp的backward
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    optimizer.step()
```

#### 1.3.3 opt_level详解

| Level | 模式 | 说明 | 适用场景 |
|-------|-----|------|---------|
| O0 | FP32 | 全精度训练 | Debug, baseline |
| O1 | 混合精度 | 白名单算子FP16 | 通用推荐 |
| O2 | 几乎全FP16 | 除BN/LN外全FP16 | 激进优化 |
| O3 | 全FP16 | 全部FP16 | 不推荐（不稳定） |

**910C推荐**：O2（最佳性能/稳定性平衡）

### 1.4 MindSpore混合精度

```python
from mindspore import context, Model
from mindspore.train.amp import build_train_network

# 配置混合精度
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# 定义网络
network = YourNetwork()

# 配置混合精度
network = build_train_network(
    network,
    optimizer,
    level='O2',  # O0, O1, O2, O3
    loss_scale_manager=None,  # 自动loss scaling
    keep_batchnorm_fp32=True
)

# 训练
model = Model(network)
model.train(epoch, train_dataset)
```

### 1.5 Loss Scaling技巧

**为什么需要Loss Scaling？**
- FP16表示范围小，梯度容易下溢
- Loss scaling放大梯度，防止变成0

**动态Loss Scaling**：
```python
# 自动调整scale值
scaler = GradScaler(
    init_scale=2.**16,  # 初始scale
    growth_factor=2.0,  # 增长因子
    backoff_factor=0.5,  # 回退因子
    growth_interval=2000,  # 增长间隔
)
```

**固定Loss Scaling**：
```python
# 适合稳定的训练任务
scaler = GradScaler(
    init_scale=1024,
    growth_factor=1.0,  # 不增长
)
```

---

## 2. 算子融合与图优化

### 2.1 算子融合原理

**未融合**：
```
Input -> Conv -> [写HBM] -> [读HBM] -> BN -> [写HBM] -> [读HBM] -> ReLU -> Output
```

**融合后**：
```
Input -> Conv+BN+ReLU (一个算子) -> Output
```

**收益**：
- 减少HBM访问次数
- 提升AI Core利用率
- 降低kernel启动开销

### 2.2 常见融合模式

#### 2.2.1 垂直融合（Element-wise）

```python
# 这些操作会自动融合
x = conv(input)
x = batch_norm(x)
x = relu(x)
x = dropout(x)

# 编译后变成一个融合算子：ConvBNReluDropout
```

#### 2.2.2 水平融合

```python
# 多个独立分支并行执行
branch1 = conv1(input)
branch2 = conv2(input)
branch3 = conv3(input)

# 可能融合为一个多输出算子
```

### 2.3 PyTorch图优化

#### 2.3.1 TorchScript静态图

```python
import torch

# 方法1：torch.jit.trace
model = YourModel().npu()
example_input = torch.randn(1, 3, 224, 224).npu()
traced_model = torch.jit.trace(model, example_input)

# 方法2：torch.jit.script（支持控制流）
scripted_model = torch.jit.script(model)

# 保存
traced_model.save("model_traced.pt")

# 使用
traced_model.eval()
with torch.no_grad():
    output = traced_model(input)
```

**优化效果**：
- 算子融合：自动
- 常量折叠：是
- 死代码消除：是
- 性能提升：10-30%

#### 2.3.2 Torch-NPU特定优化

```python
import torch_npu

# 启用NPU特定优化
torch_npu.npu.set_compile_mode(jit_compile=True)

# 算子融合配置
torch_npu.npu.set_option({
    "NPU_FUZZY_COMPILE_BLACKLIST": "LayerNorm",  # 排除某些算子
    "ACL_OP_COMPILER_CACHE_MODE": "enable",  # 启用编译缓存
    "ACL_OP_COMPILER_CACHE_DIR": "./cache",
})
```

### 2.4 MindSpore自动图优化

```python
from mindspore import context

# 启用图优化
context.set_context(
    mode=context.GRAPH_MODE,  # 静态图模式（必须）
    device_target="Ascend",
    enable_graph_kernel=True,  # 算子融合
    graph_kernel_flags="--enable_parallel_fusion "  # 并行融合
                      "--enable_trans_op_optimize "  # 转置优化
                      "--enable_cluster_ops=MatMul,Add,Sub",  # 指定融合算子
)
```

**MindSpore图优化能力**：
- 算子融合：自动识别融合模式
- 算子替换：用高效算子替换低效组合
- 数据排布优化：减少transpose
- 公共子表达式消除

### 2.5 手动算子融合

对于特殊模式，可以手动定义融合算子：

```python
# PyTorch自定义融合算子
import torch
from torch.nn import functional as F

class FusedConvBNReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 这个融合会被NPU识别并优化
        return F.relu(self.bn(self.conv(x)), inplace=True)
```

---

## 3. 内存优化

### 3.1 激活重计算（Gradient Checkpointing）

**原理**：前向传播时不保存中间激活，反向传播时重新计算。

**收益**：
- 内存节省：50-80%
- 时间增加：20-30%
- **适合**：显存受限的大模型训练

#### 3.1.1 PyTorch实现

```python
import torch.utils.checkpoint as checkpoint

class TransformerBlock(torch.nn.Module):
    def forward(self, x):
        # 使用checkpoint包裹
        return checkpoint.checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        # 实际前向逻辑
        x = self.attention(x)
        x = self.ffn(x)
        return x
```

**分段策略**：
```python
# 每N层checkpoint一次
class TransformerModel(torch.nn.Module):
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % 4 == 0:  # 每4层checkpoint
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

#### 3.1.2 MindSpore重计算

```python
from mindspore import nn

class TransformerBlock(nn.Cell):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.ffn = FFN()

        # 启用重计算
        self.attention.recompute()
        self.ffn.recompute()

    def construct(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        return x
```

### 3.2 CPU Offload

**原理**：将部分数据（优化器状态、激活值）卸载到CPU内存。

**注意**：PCIe带宽有限，谨慎使用！

```python
# DeepSpeed CPU Offload配置
ds_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}
```

**910C使用建议**：
- ✅ Offload优化器状态（影响小）
- ⚠️ 谨慎offload参数（通信开销大）
- ❌ 不要offload激活值（频繁访问）

### 3.3 显存碎片整理

```python
import torch_npu

# 定期清理显存碎片
if step % 100 == 0:
    torch_npu.npu.empty_cache()

# 或在特定时机
torch_npu.npu.synchronize()
torch_npu.npu.empty_cache()
```

### 3.4 混合内存管理

```python
# 使用统一内存管理
torch_npu.npu.set_memory_strategy({
    "enable_unified_memory": True,  # 统一内存寻址
    "memory_recycle": True,  # 自动回收
})
```

---

## 4. 数据加载优化

数据加载往往是训练的隐藏瓶颈，特别是在NPU计算非常快的情况下。

### 4.1 DataLoader配置

```python
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # 关键参数
    pin_memory=True,  # 固定内存，加速传输
    prefetch_factor=2,  # 每个worker预取batch数
    persistent_workers=True,  # worker进程持久化
    drop_last=True,  # 丢弃不完整batch
)
```

**num_workers调优**：
```python
# 经验公式
num_workers = min(
    4 * num_gpus,  # 每卡4个worker
    cpu_count(),  # 不超过CPU核数
    16  # 上限16（避免进程过多）
)
```

### 4.2 数据预处理优化

#### 4.2.1 使用NPU算子

```python
import torch_npu
from torchvision import transforms

# 使用NPU加速的数据增强
npu_transforms = torch.nn.Sequential(
    torch_npu.contrib.module.NpuNormalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
    torch_npu.contrib.module.NpuRandomResizedCrop(224),
)

# 在NPU上执行
inputs = inputs.npu()
inputs = npu_transforms(inputs)
```

#### 4.2.2 DALI加速（如果支持）

```python
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops

class DaliPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id)
        # DALI算子定义
        self.input = ops.FileReader(file_root=data_path)
        self.decode = ops.ImageDecoder(device="mixed")
        self.resize = ops.Resize(device="gpu", resize_x=224, resize_y=224)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.resize(images)
        return images, labels
```

### 4.3 数据缓存

```python
# 小数据集可以预加载到内存
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.cache = [data for data in dataset]

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.cache)

# 使用
train_dataset = CachedDataset(original_dataset)
```

### 4.4 异步数据传输

```python
# 使用非阻塞传输
for batch in dataloader:
    inputs = inputs.to('npu:0', non_blocking=True)
    labels = labels.to('npu:0', non_blocking=True)

    # 开始计算（与数据传输overlap）
    outputs = model(inputs)
```

---

## 5. 优化器优化

### 5.1 融合优化器

昇腾提供融合版本的优化器，性能更好：

```python
import torch_npu

# 使用NPU融合优化器
optimizer = torch_npu.optim.NpuFusedAdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# 其他融合优化器
# - NpuFusedAdam
# - NpuFusedSGD
# - NpuFusedLamb
```

**性能提升**：5-15%

### 5.2 梯度裁剪优化

```python
# 使用融合梯度裁剪
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    norm_type=2.0
)

# 或在优化器中配置
optimizer = torch_npu.optim.NpuFusedAdamW(
    model.parameters(),
    lr=1e-4,
    clip_grad=1.0  # 内置裁剪
)
```

### 5.3 学习率调度

```python
from torch.optim.lr_scheduler import OneCycleLR

# One-cycle学习率策略
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # warmup占比
    anneal_strategy='cos'
)

# 每个batch更新
for batch in train_loader:
    ...
    optimizer.step()
    scheduler.step()
```

---

## 6. 模型结构优化

### 6.1 使用NPU友好的算子

**推荐**：
- ✅ 标准卷积（Conv2d）
- ✅ 矩阵乘法（Linear）
- ✅ LayerNorm / BatchNorm
- ✅ GELU / ReLU
- ✅ Attention（标准实现）

**谨慎使用**：
- ⚠️ 动态shape操作
- ⚠️ 复杂控制流
- ⚠️ 自定义算子

### 6.2 通道数对齐

```python
# 将通道数对齐到16的倍数（Cube单元优化）
def align_channels(channels, align=16):
    return ((channels + align - 1) // align) * align

# 模型定义
class OptimizedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        in_channels = align_channels(in_channels)
        out_channels = align_channels(out_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
```

### 6.3 批次大小优化

```python
# 找到最优batch size
def find_optimal_batch_size(model, start=32, max_size=512):
    for batch_size in [start, start*2, start*4, start*8]:
        if batch_size > max_size:
            break

        try:
            dummy_input = torch.randn(batch_size, 3, 224, 224).npu()
            _ = model(dummy_input)
            print(f"Batch size {batch_size}: OK")
        except RuntimeError as e:
            print(f"Batch size {batch_size}: OOM")
            return batch_size // 2

    return batch_size
```

---

## 7. 性能监控与调试

### 7.1 使用msprof

```bash
# 采集性能数据
msprof --application="python train.py" \
       --output=./profiling_output \
       --ai-core=on \
       --aicpu=on \
       --task-trace=on \
       --hccl-trace=on

# 分析timeline
# 在浏览器中打开：profiling_output/timeline.json
```

### 7.2 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= 5:
            break
        outputs = model(inputs)
        loss.backward()
        optimizer.step()
        prof.step()

# 查看结果
# tensorboard --logdir=./log
```

### 7.3 关键指标

```python
import torch_npu

# AI Core利用率
utilization = torch_npu.npu.utilization(device=0)
print(f"AI Core: {utilization}%")

# 内存使用
memory_allocated = torch_npu.npu.memory_allocated(0) / 1024**3  # GB
memory_reserved = torch_npu.npu.memory_reserved(0) / 1024**3
print(f"Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

# 吞吐量
samples_per_sec = batch_size * world_size / step_time
print(f"Throughput: {samples_per_sec:.2f} samples/s")
```

---

## 8. 完整训练示例

```python
import torch
import torch_npu
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

def train():
    # 1. 初始化分布式
    torch.distributed.init_process_group(backend='hccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch_npu.npu.set_device(local_rank)

    # 2. 创建模型
    model = YourModel().npu()
    model = DDP(model, device_ids=[local_rank])

    # 3. 优化器（融合版本）
    optimizer = torch_npu.optim.NpuFusedAdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    # 4. 混合精度
    scaler = GradScaler()

    # 5. 数据加载
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    # 6. 训练循环
    model.train()
    gradient_accumulation_steps = 4

    for epoch in range(num_epochs):
        for step, (inputs, labels) in enumerate(train_loader):
            # 异步传输
            inputs = inputs.npu(non_blocking=True)
            labels = labels.npu(non_blocking=True)

            # 混合精度前向
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps

            # 梯度缩放反向
            scaler.scale(loss).backward()

            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 日志
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")

    # 7. 保存模型
    if local_rank == 0:
        torch.save(model.module.state_dict(), 'model.pth')

if __name__ == '__main__':
    train()
```

---

## 9. 常见问题排查

### 问题1：训练速度慢

**检查清单**：
1. AI Core利用率是否 >80%？
   - 否 -> 增大batch size / 优化数据加载
2. 是否使用混合精度？
   - 否 -> 启用FP16/BF16
3. 数据加载是否瓶颈？
   - 是 -> 增加num_workers / 使用数据缓存
4. 是否有频繁的CPU-NPU传输？
   - 是 -> 减少.cpu()/.item()调用

### 问题2：显存溢出

**解决方案**：
1. 减小batch size
2. 启用gradient checkpointing
3. 使用ZeRO Stage 2
4. 减小模型规模

### 问题3：数值不稳定

**解决方案**：
1. 使用BF16替代FP16
2. 调整loss scaling参数
3. 检查学习率是否过大
4. 确保BatchNorm/LayerNorm使用FP32

---

## 10. 性能优化检查清单

训练前确保以下优化已启用：

**必须项**：
- [ ] 混合精度训练（FP16/BF16）
- [ ] 数据并行（多卡）
- [ ] 梯度累积（PCIe场景）
- [ ] 融合优化器

**推荐项**：
- [ ] 静态图编译（TorchScript / MindSpore图模式）
- [ ] 算子融合
- [ ] 通信计算重叠
- [ ] 数据加载优化（num_workers, pin_memory）

**可选项**：
- [ ] Gradient Checkpointing（显存受限时）
- [ ] ZeRO优化（大模型）
- [ ] 梯度压缩（通信受限时）

---

## 11. 总结

昇腾910C训练优化的核心要点：

1. **混合精度必备**：不用混合精度就是浪费
2. **数据并行优先**：最适合PCIe架构
3. **充分利用编译优化**：静态图 + 算子融合
4. **内存管理**：合理使用重计算和ZeRO
5. **数据加载不能忽视**：避免成为瓶颈

遵循这些优化原则，910C可以达到与NVIDIA GPU相当的训练效率。
