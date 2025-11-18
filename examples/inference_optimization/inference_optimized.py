#!/usr/bin/env python3
"""
昇腾910C优化推理示例
展示了TorchScript、FP16、批处理等推理优化技术
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch_npu
from typing import List, Tuple


class ResNetBlock(nn.Module):
    """ResNet基础块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    """简化的ResNet模型"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class OptimizedInference:
    """优化的推理类"""

    def __init__(self, model_path=None, use_fp16=True, use_jit=True):
        """
        Args:
            model_path: 模型路径（如果为None则创建新模型）
            use_fp16: 是否使用FP16推理
            use_jit: 是否使用TorchScript
        """
        self.use_fp16 = use_fp16
        self.use_jit = use_jit

        # 创建或加载模型
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = SimpleResNet()

        # 移到NPU并设置为评估模式
        self.model = self.model.npu().eval()

        # FP16优化
        if self.use_fp16:
            self.model = self.model.half()

        # TorchScript优化
        if self.use_jit:
            self._optimize_with_jit()

        # Warmup
        self._warmup()

    def _optimize_with_jit(self):
        """使用TorchScript优化模型"""
        print("Optimizing model with TorchScript...")

        # 创建示例输入
        dtype = torch.float16 if self.use_fp16 else torch.float32
        example_input = torch.randn(1, 3, 224, 224, dtype=dtype).npu()

        # Trace模型
        self.model = torch.jit.trace(self.model, example_input)

        # 冻结模型（移除训练相关操作）
        self.model = torch.jit.freeze(self.model)

        # 进一步优化
        self.model = torch.jit.optimize_for_inference(self.model)

        print("TorchScript optimization completed")

    def _warmup(self, iterations=10):
        """Warmup推理（触发编译）"""
        print(f"Warming up for {iterations} iterations...")

        dtype = torch.float16 if self.use_fp16 else torch.float32
        dummy_input = torch.randn(1, 3, 224, 224, dtype=dtype).npu()

        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)

        torch_npu.npu.synchronize()
        print("Warmup completed")

    @torch.no_grad()
    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        """单次推理"""
        # 确保数据类型正确
        if self.use_fp16:
            inputs = inputs.half()

        # 推理
        outputs = self.model(inputs)
        return outputs

    @torch.no_grad()
    def infer_batch(self, inputs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """批量推理"""
        # 堆叠为batch
        batch_inputs = torch.stack(inputs_list)

        # 推理
        batch_outputs = self.infer(batch_inputs)

        # 拆分
        return list(batch_outputs)

    def benchmark_latency(self, batch_size=1, iterations=1000) -> dict:
        """测试延迟"""
        print(f"\n=== Latency Benchmark (batch_size={batch_size}) ===")

        dtype = torch.float16 if self.use_fp16 else torch.float32
        test_input = torch.randn(batch_size, 3, 224, 224, dtype=dtype).npu()

        latencies = []

        for _ in range(iterations):
            torch_npu.npu.synchronize()
            start = time.time()

            _ = self.model(test_input)

            torch_npu.npu.synchronize()
            latencies.append((time.time() - start) * 1000)  # ms

        latencies = np.array(latencies)

        results = {
            'batch_size': batch_size,
            'iterations': iterations,
            'p50': np.percentile(latencies, 50),
            'p90': np.percentile(latencies, 90),
            'p99': np.percentile(latencies, 99),
            'mean': np.mean(latencies),
            'std': np.std(latencies),
        }

        print(f"Results:")
        print(f"  P50: {results['p50']:.2f} ms")
        print(f"  P90: {results['p90']:.2f} ms")
        print(f"  P99: {results['p99']:.2f} ms")
        print(f"  Mean: {results['mean']:.2f} ms")
        print(f"  Std: {results['std']:.2f} ms")

        return results

    def benchmark_throughput(self, batch_size=32, duration=60) -> dict:
        """测试吞吐量"""
        print(f"\n=== Throughput Benchmark (batch_size={batch_size}) ===")

        dtype = torch.float16 if self.use_fp16 else torch.float32
        test_input = torch.randn(batch_size, 3, 224, 224, dtype=dtype).npu()

        iterations = 0
        torch_npu.npu.synchronize()
        start_time = time.time()

        while time.time() - start_time < duration:
            _ = self.model(test_input)
            iterations += 1

        torch_npu.npu.synchronize()
        elapsed = time.time() - start_time

        throughput = (iterations * batch_size) / elapsed

        results = {
            'batch_size': batch_size,
            'duration': duration,
            'iterations': iterations,
            'throughput': throughput,
        }

        print(f"Results:")
        print(f"  Duration: {elapsed:.2f} s")
        print(f"  Iterations: {iterations}")
        print(f"  Throughput: {throughput:.2f} samples/s")

        return results

    def find_optimal_batch_size(self, max_batch=256) -> Tuple[int, float]:
        """寻找最优batch size"""
        print(f"\n=== Finding Optimal Batch Size ===")

        best_throughput = 0
        best_batch = 1

        dtype = torch.float16 if self.use_fp16 else torch.float32

        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if batch_size > max_batch:
                break

            try:
                # 测试此batch size
                test_input = torch.randn(batch_size, 3, 224, 224, dtype=dtype).npu()

                # 预热
                for _ in range(10):
                    _ = self.model(test_input)

                # 测量
                iterations = 100
                torch_npu.npu.synchronize()
                start = time.time()

                for _ in range(iterations):
                    _ = self.model(test_input)

                torch_npu.npu.synchronize()
                elapsed = time.time() - start

                throughput = (iterations * batch_size) / elapsed

                print(f"Batch size {batch_size:3d}: {throughput:8.2f} samples/s")

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch = batch_size

            except RuntimeError as e:
                print(f"Batch size {batch_size:3d}: OOM")
                break

        print(f"\nOptimal batch size: {best_batch} ({best_throughput:.2f} samples/s)")
        return best_batch, best_throughput


def main():
    parser = argparse.ArgumentParser(description='910C Inference Optimization Example')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    parser.add_argument('--no_fp16', action='store_true', help='Disable FP16')
    parser.add_argument('--no_jit', action='store_true', help='Disable TorchScript')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--find_optimal_batch', action='store_true',
                        help='Find optimal batch size')
    args = parser.parse_args()

    print("=== 910C Inference Optimization Example ===")
    print(f"FP16: {not args.no_fp16}")
    print(f"TorchScript: {not args.no_jit}")

    # 创建优化推理实例
    inference = OptimizedInference(
        model_path=args.model_path,
        use_fp16=not args.no_fp16,
        use_jit=not args.no_jit
    )

    # 测试单次推理
    print("\n=== Single Inference Test ===")
    dtype = torch.float16 if not args.no_fp16 else torch.float32
    test_input = torch.randn(1, 3, 224, 224, dtype=dtype).npu()

    start = time.time()
    output = inference.infer(test_input)
    elapsed = time.time() - start

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Inference time: {elapsed*1000:.2f} ms")

    # 运行benchmark
    if args.benchmark:
        # 延迟测试
        inference.benchmark_latency(batch_size=1, iterations=1000)
        inference.benchmark_latency(batch_size=32, iterations=1000)

        # 吞吐量测试
        inference.benchmark_throughput(batch_size=32, duration=30)

    # 寻找最优batch size
    if args.find_optimal_batch:
        inference.find_optimal_batch_size(max_batch=256)


if __name__ == '__main__':
    main()
