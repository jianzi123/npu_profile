# 华为昇腾910C性能调优指南

本项目提供华为昇腾910C NPU的全面性能调优指南，涵盖硬件特性、并行策略、训练优化和推理优化。

## 📋 目录

### 实践指南
- [910C硬件特性](docs/910c_features.md)
- [并行调优指南](docs/parallel_tuning.md)
- [训练优化](docs/training_optimization.md)
- [推理优化](docs/inference_optimization.md)

### 深度分析（理论与建模）
- [架构深度剖析](docs/deep_dive_architecture.md) - AI Core微架构、内存层次、PCIe互连深度分析
- [并行性能建模](docs/deep_dive_parallel_performance.md) - 数学建模、扩展性理论、通信优化
- [量化算法原理](docs/deep_dive_quantization.md) - 量化理论、QAT/PTQ深入、硬件加速原理

### 案例分析
- [Qwen3-30B完整优化流程](docs/case_study_qwen3_30b.md) - 从模型分析到部署的端到端案例

## 🎯 项目目标

帮助开发者和研究人员充分利用昇腾910C的硬件能力，针对其PCIe架构特点进行优化，实现最佳的训练和推理性能。

## 📚 文档结构

```
npu_profile/
├── README.md                                    # 项目概述
├── docs/
│   ├── 910c_features.md                        # 910C硬件特性详解
│   ├── parallel_tuning.md                      # 并行调优策略
│   ├── training_optimization.md                # 训练优化技巧
│   ├── inference_optimization.md               # 推理优化方法
│   ├── deep_dive_architecture.md               # 架构深度剖析（理论）
│   ├── deep_dive_parallel_performance.md       # 并行性能建模（数学）
│   └── deep_dive_quantization.md               # 量化算法原理（算法）
├── examples/
│   ├── distributed_training/                   # 分布式训练示例
│   │   ├── train_ddp.py                       # DDP训练脚本
│   │   └── run_8npu.sh                        # 8卡启动脚本
│   ├── inference_optimization/                 # 推理优化示例
│   │   └── inference_optimized.py             # 优化推理代码
│   └── configs/                                # 配置文件示例
│       ├── training_config.yaml               # 训练配置
│       └── inference_config.yaml              # 推理配置
└── benchmarks/                                 # 性能测试脚本
    ├── benchmark_communication.py              # 通信性能测试
    └── README.md
```

## 🚀 快速开始

### 环境要求

- 昇腾910C NPU
- CANN (Compute Architecture for Neural Networks) 版本 >= 6.0
- PyTorch + Torch-NPU 或 MindSpore
- Python >= 3.8

### 核心优化方向

1. **硬件特性理解**：了解910C的PCIe架构、内存层次、AI Core特性
2. **通信优化**：针对PCIe带宽的并行策略调整
3. **算子融合**：利用昇腾的算子融合能力减少内存访问
4. **混合精度**：充分利用FP16/BF16加速能力

## 📖 主要内容概览

### 1. 910C特性
- Ascend 910C硬件架构
- PCIe连接特点vs NVLink
- AI Core和向量计算单元
- 内存体系和带宽特性

### 2. 并行调优
- PCIe架构下的数据并行策略
- 模型并行和流水线并行配置
- HCCL集合通信优化
- 梯度累积和通信重叠

### 3. 训练优化
- 混合精度训练
- 算子融合和图优化
- 内存优化（重计算、offload等）
- 数据加载和预处理优化

### 4. 推理优化
- 模型量化（INT8/INT4）
- 静态图和动态shape优化
- 批处理优化
- ATC模型转换最佳实践

## 🔧 工具链

- **CANN**: 昇腾异构计算架构
- **Torch-NPU**: PyTorch适配层
- **MindSpore**: 华为深度学习框架
- **ATC**: 模型转换工具
- **Profiling工具**: 性能分析工具（msprof、PyTorch Profiler）

## 📊 性能对比

文档中包含了针对不同场景的性能基准测试和优化前后对比。

## 🎓 深度分析特色

本项目不仅提供实践指南，还包含三份深度技术分析文档：

### 1. 架构深度剖析
- **AI Core微架构**：Cube/Vector/Scalar单元的流水线、吞吐量分析
- **内存层次详解**：HBM带宽测量、Cache行为、NUMA效应
- **Roofline模型**：性能上界分析、计算与内存bound判断
- **算子执行模型**：编译器优化、算子融合收益量化
- **关键性能陷阱**：隐式同步点、假共享、不规则访问

### 2. 并行性能建模
- **扩展性理论**：Amdahl定律、通信模型（α-β模型）
- **梯度累积数学优化**：最优累积步数推导
- **Ring AllReduce分析**：通信时间公式、效率损失原因
- **3D并行最优配置**：DP/TP/PP组合的数学建模
- **ZeRO内存-通信权衡**：Stage 1-3的精确分析
- **性能预测模型**：端到端训练时间预测公式

### 3. 量化算法原理
- **量化数学基础**：SQNR公式、量化噪声分析
- **QAT原理**：STE（直通估计器）、可学习量化参数
- **PTQ算法**：Min-Max/Percentile/MSE/KL散度方法对比
- **混合精度搜索**：敏感度分析、贪心/进化算法
- **INT8/INT4硬件加速**：Cube单元实现、数值精度分析
- **GPTQ算法**：大模型量化的最优脑量化

**适用人群**：
- 实践指南：工程师、算法开发者
- 深度分析：系统架构师、研究人员、需要深度优化的专家

## 🤝 贡献

欢迎提交优化建议和最佳实践案例。

## 📝 许可证

MIT License

## 📮 联系方式

如有问题或建议，请提交Issue。
