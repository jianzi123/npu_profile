# 华为昇腾910C性能调优指南

本项目提供华为昇腾910C NPU的全面性能调优指南，涵盖硬件特性、并行策略、训练优化和推理优化。

## 📋 目录

- [910C硬件特性](docs/910c_features.md)
- [并行调优指南](docs/parallel_tuning.md)
- [训练优化](docs/training_optimization.md)
- [推理优化](docs/inference_optimization.md)

## 🎯 项目目标

帮助开发者和研究人员充分利用昇腾910C的硬件能力，针对其PCIe架构特点进行优化，实现最佳的训练和推理性能。

## 📚 文档结构

```
npu_profile/
├── README.md                           # 项目概述
├── docs/
│   ├── 910c_features.md               # 910C硬件特性详解
│   ├── parallel_tuning.md             # 并行调优策略
│   ├── training_optimization.md       # 训练优化技巧
│   └── inference_optimization.md      # 推理优化方法
├── examples/
│   ├── distributed_training/          # 分布式训练示例
│   ├── inference_optimization/        # 推理优化示例
│   └── configs/                       # 配置文件示例
└── benchmarks/                        # 性能测试脚本
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

## 🤝 贡献

欢迎提交优化建议和最佳实践案例。

## 📝 许可证

MIT License

## 📮 联系方式

如有问题或建议，请提交Issue。
