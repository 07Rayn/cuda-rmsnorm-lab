
# CUDA RMSNorm Lab
GPU算子开发与性能优化项目,围绕Transformeer中的RMSNorm和残差融合计算展开.
## 项目目标

使用 CUDA C++ 实现 RMSNorm 与 Add + RMSNorm 算子，通过并行归约、向量化访存和线程组织优化，探索不同输入形状下的性能表现。

## 开发内容

- 实现基础 CUDA RMSNorm，并接入 PyTorch。
- 优化线程分工、归约策略及数据加载方式。
- 实现 Residual Add 与 RMSNorm 融合。
- 与 PyTorch、Liger-Kernel 进行正确性及性能对比。
- 使用 Nsight Compute 分析性能瓶颈，记录优化过程与实验结果。

## 技术栈

CUDA C++、Python、PyTorch、Triton、Git、Nsight Compute。

## 当前进度

项目初始化阶段，正在建立代码结构和验证流程，尚未完成 CUDA 算子实现及性能优化。
