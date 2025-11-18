#!/usr/bin/env python3
"""
昇腾910C分布式训练示例（PyTorch DDP）
展示了数据并行、混合精度、梯度累积等优化技术
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler


def setup_distributed():
    """初始化分布式环境"""
    # 从环境变量获取rank信息
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 初始化进程组（使用HCCL后端）
    dist.init_process_group(
        backend='hccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # 设置当前设备
    torch_npu.npu.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    """简单的示例模型"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_dataloader(batch_size, world_size, rank):
    """创建数据加载器"""
    # 这里使用随机数据作为示例
    # 实际使用时应该加载真实数据集
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    # 生成随机数据
    num_samples = 10000
    data = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_samples,))
    dataset = TensorDataset(data, labels)

    # 使用DistributedSampler确保数据不重复
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    return dataloader


def train_epoch(model, dataloader, criterion, optimizer, scaler,
                epoch, rank, args):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_samples = 0
    start_time = time.time()

    for step, (inputs, labels) in enumerate(dataloader):
        # 异步传输到NPU
        inputs = inputs.npu(non_blocking=True)
        labels = labels.npu(non_blocking=True)

        # 混合精度前向传播
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 梯度累积缩放
            loss = loss / args.gradient_accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 统计
        total_loss += loss.item() * args.gradient_accumulation_steps
        total_samples += inputs.size(0)

        # 日志输出（仅rank 0）
        if rank == 0 and step % args.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            samples_per_sec = total_samples / (time.time() - start_time)
            print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Throughput: {samples_per_sec:.2f} samples/s")

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='910C Distributed Training Example')
    parser.add_argument('--batch_size', type=int, default=32, help='Per-device batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Model save path')
    args = parser.parse_args()

    # 初始化分布式
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print(f"Starting distributed training on {world_size} NPUs")
        print(f"Configuration:")
        print(f"  - Batch size per device: {args.batch_size}")
        print(f"  - Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  - Learning rate: {args.lr}")

    # 创建模型
    model = SimpleModel().npu()

    # DDP包装
    model = DDP(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=25,  # PCIe场景优化
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
        find_unused_parameters=False
    )

    # 创建优化器（使用融合优化器）
    optimizer = torch_npu.optim.NpuFusedAdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # 混合精度scaler
    scaler = GradScaler()

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 数据加载器
    train_loader = get_dataloader(args.batch_size, world_size, rank)

    # 训练循环
    for epoch in range(args.epochs):
        # 设置epoch（用于DistributedSampler的shuffle）
        train_loader.sampler.set_epoch(epoch)

        # 训练一个epoch
        avg_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, rank, args
        )

        if rank == 0:
            print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")

            # 保存checkpoint
            if (epoch + 1) % 5 == 0:
                os.makedirs(args.save_path, exist_ok=True)
                checkpoint_path = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    # 清理
    cleanup()

    if rank == 0:
        print("Training completed!")


if __name__ == '__main__':
    main()
