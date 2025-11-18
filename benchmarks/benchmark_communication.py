#!/usr/bin/env python3
"""
HCCL通信性能测试脚本
测试不同消息大小下的AllReduce带宽和延迟
"""

import os
import time
import argparse
import torch
import torch.distributed as dist
import torch_npu


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    dist.init_process_group(
        backend='hccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch_npu.npu.set_device(local_rank)
    return rank, local_rank, world_size


def benchmark_allreduce(size_mb, iterations=100, dtype=torch.float32):
    """
    测试AllReduce性能

    Args:
        size_mb: 消息大小(MB)
        iterations: 迭代次数
        dtype: 数据类型
    """
    # 计算tensor大小
    bytes_per_element = 4 if dtype == torch.float32 else 2
    tensor_size = int(size_mb * 1024 * 1024 / bytes_per_element)

    # 创建tensor
    tensor = torch.randn(tensor_size, dtype=dtype).npu()

    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 测量
    torch_npu.npu.synchronize()
    start_time = time.time()

    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch_npu.npu.synchronize()
    elapsed = time.time() - start_time

    # 计算指标
    # AllReduce的算法带宽 = 2 * (N-1) / N * data_size / time
    # 其中N是world_size
    world_size = dist.get_world_size()
    algo_bandwidth = 2 * (world_size - 1) / world_size * size_mb * iterations / elapsed

    # Bus带宽（实际物理带宽）
    bus_bandwidth = algo_bandwidth * world_size / (2 * (world_size - 1))

    # 平均延迟
    avg_latency = (elapsed / iterations) * 1000  # ms

    return {
        'size_mb': size_mb,
        'algo_bandwidth': algo_bandwidth,
        'bus_bandwidth': bus_bandwidth,
        'avg_latency': avg_latency,
        'iterations': iterations
    }


def main():
    parser = argparse.ArgumentParser(description='HCCL Communication Benchmark')
    parser.add_argument('--dtype', type=str, default='fp32',
                        choices=['fp32', 'fp16'], help='Data type')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations')
    args = parser.parse_args()

    # 初始化分布式
    rank, local_rank, world_size = setup_distributed()

    dtype = torch.float32 if args.dtype == 'fp32' else torch.float16

    if rank == 0:
        print("=" * 80)
        print("HCCL AllReduce Bandwidth Benchmark")
        print("=" * 80)
        print(f"World size: {world_size}")
        print(f"Data type: {args.dtype}")
        print(f"Iterations: {args.iterations}")
        print("=" * 80)
        print(f"{'Size (MB)':<12} {'Algo BW (MB/s)':<18} {'Bus BW (MB/s)':<18} {'Latency (ms)':<15}")
        print("-" * 80)

    # 测试不同消息大小
    test_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for size_mb in test_sizes:
        try:
            results = benchmark_allreduce(
                size_mb,
                iterations=args.iterations,
                dtype=dtype
            )

            if rank == 0:
                print(f"{results['size_mb']:<12.1f} "
                      f"{results['algo_bandwidth']:<18.2f} "
                      f"{results['bus_bandwidth']:<18.2f} "
                      f"{results['avg_latency']:<15.3f}")

        except RuntimeError as e:
            if rank == 0:
                print(f"{size_mb:<12.1f} OOM")
            break

    # 清理
    dist.destroy_process_group()

    if rank == 0:
        print("=" * 80)
        print("Benchmark completed!")
        print("\n说明:")
        print("- Algo BW: 算法带宽（考虑了AllReduce的通信模式）")
        print("- Bus BW: 总线带宽（实际物理带宽）")
        print("- 对于PCIe 4.0 x16，理论单向带宽约为32 GB/s")
        print("=" * 80)


if __name__ == '__main__':
    main()
