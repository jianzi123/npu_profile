# 昇腾910C推理优化指南

推理优化与训练优化有很大不同，重点在于降低延迟、提高吞吐量和减少资源消耗。本文档涵盖910C上的各种推理优化技术。

---

## 1. 推理vs训练的差异

| 维度 | 训练 | 推理 |
|-----|-----|-----|
| 计算模式 | 前向+反向 | 仅前向 |
| 内存需求 | 高（需保存中间激活） | 低 |
| batch size | 尽可能大 | 动态变化 |
| 精度 | FP16/BF16 | INT8/INT4 |
| 延迟要求 | 不敏感 | 高度敏感 |
| 吞吐量 | 重要 | 非常重要 |

**推理优化目标**：
1. 降低延迟（latency）
2. 提高吞吐量（throughput）
3. 减少显存占用
4. 降低功耗

---

## 2. 模型量化

### 2.1 量化原理

**量化**：将FP32/FP16权重和激活转换为低精度整数（INT8/INT4）。

**性能提升**：
- INT8: 4-8x 吞吐量提升
- INT4: 8-16x 吞吐量提升
- 显存占用：减少75-87.5%

### 2.2 量化训练（QAT）

**推荐方法**：训练时模拟量化，准确度最高。

```python
# PyTorch量化训练
import torch
import torch.quantization as quant

# 1. 定义量化配置
model = YourModel()
model.qconfig = quant.get_default_qat_qconfig('qnnpack')

# 2. 插入伪量化节点
model_prepared = quant.prepare_qat(model, inplace=False)

# 3. 训练（带量化感知）
for epoch in range(num_epochs):
    train(model_prepared)

# 4. 转换为真实量化模型
model_quantized = quant.convert(model_prepared, inplace=False)

# 5. 保存
torch.save(model_quantized.state_dict(), 'model_int8.pth')
```

### 2.3 训练后量化（PTQ）

**适用场景**：无法重新训练，或数据有限。

```python
import torch.quantization as quant

# 1. 设置量化配置
model.eval()
model.qconfig = quant.get_default_qconfig('qnnpack')

# 2. 插入观察者
model_prepared = quant.prepare(model, inplace=False)

# 3. 校准（使用代表性数据）
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

# 4. 转换
model_quantized = quant.convert(model_prepared, inplace=False)
```

### 2.4 昇腾量化工具（AMCT）

**AMCT** (Ascend Model Compression Toolkit) 是华为提供的量化工具。

```python
import amct_pytorch as amct

# 1. 创建量化配置
config = {
    "calibration": {
        "batch_num": 1,
        "skip_layers": []  # 跳过某些层
    },
    "quantization": {
        "weight": {
            "mode": "SYMMETRIC",  # 对称量化
            "bit_width": 8
        },
        "activation": {
            "mode": "ASYMMETRIC",  # 非对称量化
            "bit_width": 8
        }
    }
}

# 2. 创建量化模型
model = amct.create_quant_model(
    model,
    config,
    record_file="./outputs/record.txt"
)

# 3. 校准
model.eval()
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)

# 4. 保存量化模型
amct.save_quant_model(
    model,
    save_path="./outputs/quantized_model"
)
```

### 2.5 量化精度对比

| 量化方法 | 准确度保持 | 实现难度 | 推荐场景 |
|---------|----------|---------|---------|
| QAT | 99-100% | 高（需重训练） | 关键应用 |
| PTQ | 95-98% | 低 | 快速部署 |
| Dynamic Quant | 97-99% | 最低 | 权重量化 |

---

## 3. 模型转换与ATC

### 3.1 ATC工具介绍

**ATC** (Ascend Tensor Compiler) 将训练框架模型转换为昇腾OM格式。

### 3.2 ONNX转OM

```bash
# 1. PyTorch导出ONNX
python export_onnx.py

# 2. ONNX转OM
atc --model=model.onnx \
    --framework=5 \
    --output=model \
    --input_shape="input:1,3,224,224" \
    --soc_version=Ascend910C \
    --precision_mode=allow_fp32_to_fp16 \
    --op_select_implmode=high_performance \
    --fusion_switch_file=fusion_switch.cfg
```

**关键参数**：

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `--precision_mode` | 精度模式 | `allow_fp32_to_fp16` |
| `--op_select_implmode` | 算子选择模式 | `high_performance` |
| `--input_shape` | 输入shape | 根据实际 |
| `--dynamic_dims` | 动态shape | 多档位 |
| `--fusion_switch_file` | 算子融合配置 | 可选 |

### 3.3 动态shape支持

```bash
# 支持多个batch size
atc --model=model.onnx \
    --framework=5 \
    --output=model_dynamic \
    --input_shape="input:-1,3,224,224" \
    --dynamic_dims="1;4;8;16;32" \  # 支持这些batch size
    --soc_version=Ascend910C
```

### 3.4 算子融合配置

创建`fusion_switch.cfg`：

```ini
# 启用特定融合模式
[on]
Conv2D+BiasAdd+Relu
MatMul+BiasAdd
LayerNorm

[off]
# 禁用某些融合（如果有问题）
```

---

## 4. 静态图优化

### 4.1 TorchScript导出

```python
import torch

# 方法1：trace（推荐）
model = YourModel().eval().npu()
example_input = torch.randn(1, 3, 224, 224).npu()

# Trace模型
traced_model = torch.jit.trace(model, example_input)

# 优化
traced_model = torch.jit.optimize_for_inference(traced_model)

# 保存
traced_model.save("model_traced.pt")

# 方法2：script（支持控制流）
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### 4.2 图优化

```python
import torch_npu

# 启用图优化
torch_npu.npu.set_compile_mode(jit_compile=True)

# 冻结模型（消除训练相关操作）
model.eval()
frozen_model = torch.jit.freeze(traced_model)

# 进一步优化
optimized_model = torch.jit.optimize_for_inference(
    frozen_model,
    other_optimizations=[
        "remove_dropout",
        "fuse_conv_bn",
        "constant_propagation"
    ]
)
```

---

## 5. 批处理优化

### 5.1 动态Batching

```python
class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, timeout_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue = []
        self.lock = threading.Lock()

    def infer(self, input_data):
        # 添加到队列
        future = Future()
        with self.lock:
            self.queue.append((input_data, future))

        # 等待批处理
        return future.result()

    def batch_process(self):
        while True:
            time.sleep(self.timeout_ms / 1000)

            with self.lock:
                if not self.queue:
                    continue

                # 取出batch
                batch = self.queue[:self.max_batch_size]
                self.queue = self.queue[self.max_batch_size:]

            # 批量推理
            inputs = torch.stack([item[0] for item in batch])
            outputs = self.model(inputs)

            # 返回结果
            for i, (_, future) in enumerate(batch):
                future.set_result(outputs[i])
```

### 5.2 最优batch size搜索

```python
def find_optimal_batch_size(model, input_shape, max_batch=128):
    best_throughput = 0
    best_batch = 1

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        if batch_size > max_batch:
            break

        try:
            # 测试
            dummy_input = torch.randn(batch_size, *input_shape).npu()

            # 预热
            for _ in range(10):
                _ = model(dummy_input)

            # 测量
            torch_npu.npu.synchronize()
            start = time.time()

            iterations = 100
            for _ in range(iterations):
                _ = model(dummy_input)

            torch_npu.npu.synchronize()
            elapsed = time.time() - start

            throughput = (batch_size * iterations) / elapsed

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch = batch_size

            print(f"Batch {batch_size}: {throughput:.2f} samples/s")

        except RuntimeError:
            break

    return best_batch, best_throughput
```

---

## 6. 延迟优化

### 6.1 消除warmup时间

```python
# 第一次推理会触发编译，预先warmup
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).npu()

# Warmup（触发编译）
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# 实际推理
torch_npu.npu.synchronize()
start = time.time()
output = model(real_input)
torch_npu.npu.synchronize()
latency = time.time() - start
```

### 6.2 使用流（Stream）

```python
import torch_npu

# 创建多个流
streams = [torch_npu.npu.Stream() for _ in range(4)]

def parallel_infer(inputs_list):
    results = []

    for i, inputs in enumerate(inputs_list):
        stream = streams[i % len(streams)]

        with torch_npu.npu.stream(stream):
            output = model(inputs)
            results.append(output)

    # 同步所有流
    for stream in streams:
        stream.synchronize()

    return results
```

### 6.3 降低数据传输开销

```python
# 预先分配设备内存
input_buffer = torch.empty(batch_size, 3, 224, 224).npu()
output_buffer = torch.empty(batch_size, num_classes).npu()

def fast_infer(cpu_inputs):
    # 使用预分配的buffer
    input_buffer.copy_(cpu_inputs, non_blocking=True)

    # 推理
    with torch.no_grad():
        output_buffer = model(input_buffer)

    return output_buffer
```

---

## 7. 内存优化

### 7.1 权重共享

```python
# 多个实例共享权重
class SharedModel:
    _shared_weights = None

    def __init__(self):
        if SharedModel._shared_weights is None:
            SharedModel._shared_weights = load_weights()

        self.weights = SharedModel._shared_weights
```

### 7.2 模型裁剪

```python
import torch.nn.utils.prune as prune

# 结构化剪枝
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(
            module,
            name="weight",
            amount=0.3,  # 剪枝30%
            n=2,
            dim=0
        )

# 移除剪枝重参数化
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')
```

### 7.3 知识蒸馏

```python
# 使用小模型（student）学习大模型（teacher）
teacher_model = LargeModel().eval().npu()
student_model = SmallModel().train().npu()

def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    # KL散度
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    # 硬标签损失
    ce_loss = F.cross_entropy(student_logits, labels)

    # 组合
    return alpha * kd_loss + (1 - alpha) * ce_loss

# 训练student
for batch in train_loader:
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)

    student_logits = student_model(inputs)
    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
```

---

## 8. 推理服务部署

### 8.1 MindX SDK推理

```python
from mindx import sdk

# 1. 初始化SDK
sdk.init("device_id", 0)

# 2. 加载模型
model_id = sdk.load_model("model.om")

# 3. 创建推理服务
infer_service = sdk.InferService(model_id)

# 4. 推理
def infer(image):
    # 预处理
    input_data = preprocess(image)

    # 推理
    output = infer_service.infer([input_data])

    # 后处理
    result = postprocess(output)
    return result

# 5. 销毁
sdk.unload_model(model_id)
```

### 8.2 Triton推理服务器

```python
# config.pbtxt
name: "ascend_model"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 1000
}
```

### 8.3 FastAPI服务示例

```python
from fastapi import FastAPI, File, UploadFile
import torch
import torch_npu

app = FastAPI()

# 加载模型（启动时）
model = torch.jit.load("model_traced.pt")
model.eval().npu()

# Warmup
dummy_input = torch.randn(1, 3, 224, 224).npu()
for _ in range(10):
    _ = model(dummy_input)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取图像
    image = Image.open(file.file)

    # 预处理
    input_tensor = preprocess(image).unsqueeze(0).npu()

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理
    result = postprocess(output)

    return {"prediction": result}

# 启动
# uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 9. 性能分析

### 9.1 延迟分析

```python
import time
import numpy as np

def benchmark_latency(model, input_tensor, iterations=1000):
    latencies = []

    # Warmup
    for _ in range(100):
        _ = model(input_tensor)

    # 测量
    for _ in range(iterations):
        torch_npu.npu.synchronize()
        start = time.time()

        _ = model(input_tensor)

        torch_npu.npu.synchronize()
        latencies.append((time.time() - start) * 1000)  # ms

    latencies = np.array(latencies)

    print(f"P50 latency: {np.percentile(latencies, 50):.2f} ms")
    print(f"P90 latency: {np.percentile(latencies, 90):.2f} ms")
    print(f"P99 latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"Average: {np.mean(latencies):.2f} ms")
```

### 9.2 吞吐量测试

```python
def benchmark_throughput(model, batch_size, duration=60):
    input_tensor = torch.randn(batch_size, 3, 224, 224).npu()

    # Warmup
    for _ in range(10):
        _ = model(input_tensor)

    # 测量
    torch_npu.npu.synchronize()
    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration:
        _ = model(input_tensor)
        iterations += 1

    torch_npu.npu.synchronize()
    elapsed = time.time() - start_time

    throughput = (iterations * batch_size) / elapsed
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Batch size: {batch_size}, Iterations: {iterations}")
```

---

## 10. 多模型推理

### 10.1 模型ensemble

```python
class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # 平均或投票
        return torch.mean(torch.stack(outputs), dim=0)

# 使用
ensemble = EnsembleModel([model1, model2, model3]).npu()
ensemble.eval()
```

### 10.2 多模型并行

```python
class MultiModelInference:
    def __init__(self, models, device_ids):
        self.models = []
        for model, device_id in zip(models, device_ids):
            self.models.append(model.to(f'npu:{device_id}'))

    def parallel_infer(self, inputs_list):
        results = []

        # 并行推理
        futures = []
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            for model, inputs in zip(self.models, inputs_list):
                future = executor.submit(model, inputs)
                futures.append(future)

        for future in futures:
            results.append(future.result())

        return results
```

---

## 11. 典型场景优化案例

### 11.1 CV分类模型（ResNet-50）

```python
# 优化配置
config = {
    "precision": "FP16",
    "batch_size": 32,
    "use_traced": True,
    "dynamic_batching": True,
}

# 模型准备
model = resnet50(pretrained=True).eval().npu()

# FP16推理
model = model.half()

# Trace
example_input = torch.randn(1, 3, 224, 224).half().npu()
traced_model = torch.jit.trace(model, example_input)
traced_model = torch.jit.optimize_for_inference(traced_model)

# 推理
@torch.no_grad()
def infer(image):
    input_tensor = preprocess(image).half().npu()
    output = traced_model(input_tensor)
    return output.argmax(dim=1)
```

**性能**：
- FP32: ~500 samples/s
- FP16: ~3000 samples/s
- **提升：6x**

### 11.2 NLP模型（BERT）

```python
# 优化配置
config = {
    "precision": "INT8",  # 量化
    "max_seq_length": 128,
    "batch_size": 64,
}

# 量化
model_quantized = quantize_model(bert_model, calibration_data)

# 动态shape
input_ids = torch.randint(0, 30000, (1, 128)).npu()
attention_mask = torch.ones(1, 128).npu()

traced_model = torch.jit.trace(
    model_quantized,
    (input_ids, attention_mask)
)

# 推理
@torch.no_grad()
def infer(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].npu()
    attention_mask = inputs['attention_mask'].npu()

    outputs = traced_model(input_ids, attention_mask)
    return outputs
```

**性能**：
- FP16: ~200 sentences/s
- INT8: ~800 sentences/s
- **提升：4x**

### 11.3 目标检测（YOLO）

```python
# 优化配置
config = {
    "precision": "FP16",
    "nms_threshold": 0.45,
    "conf_threshold": 0.25,
    "input_size": 640,
}

# 模型优化
model = YOLOv5().eval().half().npu()

# NMS优化（使用NPU算子）
def npu_nms(boxes, scores, iou_threshold):
    import torch_npu
    return torch_npu.npu.nms(
        boxes,
        scores,
        iou_threshold=iou_threshold
    )

# 推理pipeline
@torch.no_grad()
def detect(image):
    # 预处理
    input_tensor = preprocess(image).half().npu()

    # 推理
    predictions = model(input_tensor)

    # NMS
    boxes, scores, classes = npu_nms(
        predictions[:, :4],
        predictions[:, 4],
        config['nms_threshold']
    )

    return boxes, scores, classes
```

---

## 12. 推理优化检查清单

部署前确保以下优化：

**必须项**：
- [ ] 模型量化（INT8/FP16）
- [ ] 静态图转换（TorchScript/OM）
- [ ] Warmup消除首次延迟
- [ ] 批处理优化

**推荐项**：
- [ ] 算子融合
- [ ] 动态batching
- [ ] 多流并发
- [ ] 内存预分配

**可选项**：
- [ ] 模型剪枝
- [ ] 知识蒸馏
- [ ] 多模型ensemble

---

## 13. 性能基准

### 13.1 典型模型性能（单卡910C）

| 模型 | 精度 | Batch Size | 延迟 (P99) | 吞吐量 |
|-----|------|-----------|-----------|-------|
| ResNet-50 | FP16 | 32 | 12 ms | 3000 img/s |
| ResNet-50 | INT8 | 32 | 6 ms | 5500 img/s |
| BERT-Base | FP16 | 64 | 25 ms | 2800 seq/s |
| BERT-Base | INT8 | 64 | 10 ms | 6500 seq/s |
| YOLOv5-L | FP16 | 8 | 18 ms | 450 img/s |
| GPT-2 | FP16 | 16 | 35 ms | 460 token/s |

---

## 14. 常见问题

### 问题1：首次推理很慢

**原因**：图编译和算子选择在首次执行时进行。

**解决**：
- 启动时进行充分warmup（10-100次）
- 使用编译缓存（ACL_OP_COMPILER_CACHE_MODE）

### 问题2：量化后精度下降严重

**原因**：PTQ校准数据不足或不具代表性。

**解决**：
- 使用更多校准数据（>1000样本）
- 采用QAT量化
- 跳过敏感层（如最后的分类层）

### 问题3：批处理性能不理想

**原因**：batch size不是最优。

**解决**：
- 使用benchmark脚本搜索最优batch size
- 考虑动态batching

---

## 15. 总结

昇腾910C推理优化的关键要点：

1. **量化必备**：INT8可带来4-8x性能提升
2. **静态图**：TorchScript/OM转换，启用算子融合
3. **批处理**：找到最优batch size，使用动态batching
4. **延迟优化**：warmup、流并发、内存预分配
5. **服务化部署**：使用成熟的推理框架（Triton/FastAPI）

通过系统的优化，910C可以实现与NVIDIA GPU相当甚至更好的推理性能。
