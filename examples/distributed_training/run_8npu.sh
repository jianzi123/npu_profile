#!/bin/bash

# 8卡分布式训练启动脚本
# 适用于昇腾910C单机8卡场景

# 配置
export RANK_SIZE=8
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

# HCCL配置
export HCCL_ALGO=ring                    # AllReduce算法
export HCCL_GRAD_COMPRESSION=1           # 梯度压缩
export HCCL_STREAM_NUM=2                 # 通信流数量
export HCCL_BUFFSIZE=512                 # 通信buffer大小(MB)

# NPU性能优化
export COMBINED_ENABLE=1                 # 算子融合
export TASK_QUEUE_ENABLE=1              # 任务队列
export DYNAMIC_OP="ADD#MUL"             # 动态shape算子

# 日志配置
export ASCEND_SLOG_PRINT_TO_STDOUT=0    # 不输出到stdout
export ASCEND_GLOBAL_LOG_LEVEL=3        # 日志级别（3=WARNING）

# 创建日志目录
mkdir -p logs

# 清理之前的日志
rm -f logs/*.log

# 启动训练
echo "Starting 8-NPU distributed training..."
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

for((RANK_ID=0;RANK_ID<$RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID
    export LOCAL_RANK=$RANK_ID
    export WORLD_SIZE=$RANK_SIZE

    echo "Starting rank ${RANK_ID} on NPU ${RANK_ID}..."

    python train_ddp.py \
        --batch_size 32 \
        --epochs 10 \
        --lr 1e-3 \
        --gradient_accumulation_steps 4 \
        --save_path ./checkpoints \
        > logs/train_rank${RANK_ID}.log 2>&1 &
done

# 等待所有进程
wait

echo "Training completed! Check logs/ for details."
