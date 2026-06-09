# Pentacyclic Dissipative Systems / 五环耗散系统

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20533606.svg)](https://doi.org/10.5281/zenodo.20533606)

**A unified thermodynamic-information framework for low-entropy non-equilibrium dissipative structures, with an ML training diagnosis demo.**
**低熵非平衡耗散系统的热力学-信息论统一框架，含 ML 训练诊断 Demo。**

Author / 作者：唐昊东 (2026)

> *"你的 GPU 很忙，但 C3 说它没在学。五环诊断器让你在浪费算力之前停止训练。"*

---

## 快速开始

### 前置条件

- **Python 3.10+** 和 **.NET SDK 8.0+**
- 编译 C# 诊断引擎后端 PentacyclicSim（需单独克隆）

```bash
# 1. 克隆本仓库
git clone https://github.com/baiyaoseshi/pentacyclic-community.git
cd pentacyclic-community

# 2. 安装 Python 依赖
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. 克隆并编译 C# 后端（同级目录）
cd ..
git clone https://github.com/tanghaodong/pentacyclic-sim.git
cd pentacyclic-sim
dotnet build -c Release
cd ../pentacyclic-community

# 4. 启动
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，上传训练日志 CSV 即可。

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PENTACYCLIC_SIM_DIR` | C# 后端项目路径 | 自动检测 `../PentacyclicSim` |

### 演示数据

仓库自带两个演示 CSV，无需准备数据即可体验：

| 文件 | 说明 | 预期结果 |
|------|------|----------|
| [`demo/demo_cifar10_small.csv`](demo/demo_cifar10_small.csv) | CIFAR-10 小模型，40 epoch | ✅ 正常诊断，C3 裕度偏低 |
| [`demo/demo_reject.csv`](demo/demo_reject.csv) | W&B 教程随手跑，11 epoch | ❌ 数据不足，g-only 降级 |

> `demo_reject.csv` 只有 11 个 epoch 且 val 全为零——诊断器会拒绝给出结论并提示"数据不足"。这不是缺陷，是负责任的行为。

---

## Overview / 概述

本项目实现了**五环耗散系统（PDS）**框架——一个描述低熵非平衡耗散系统中能量、熵与信息耦合动力学的形式化理论。该框架识别出五个基本物理状态变量，并导出十条关系（五条正向等式 + 五条约束不等式），共同构成一个闭合的、自调节的动力学循环。

This project implements the **Pentacyclic Dissipative Systems (PDS)** framework — a formal theory describing the coupled dynamics of energy, entropy, and information in low-entropy non-equilibrium dissipative systems. The framework identifies five fundamental physical state variables and derives ten relations (five forward equalities + five constraint inequalities) that together form a closed, self-regulating dynamical cycle.

---

## Features / 功能

### 社区版（Community）

- ✅ CSV 训练日志上传
- ✅ 自动识别 g/e 列（train_acc / val_loss 等）
- ✅ ODE 参数拟合（调用 C# 后端）
- ✅ 核心参数展示（Epsilon0, Alpha）
- ✅ 拟合优度 R²
- ✅ g/e 观测 vs 拟合对比图
- ✅ g-only 模式检测与警告
- ✅ 前 1 条诊断建议

### 专业版（Professional） — [联系我们](mailto:937692907@qq.com)

- 🔒 全部 6 项拟合参数及物理含义解读
- 🔒 五变量演化轨迹分析（g, e, m, f, w）
- 🔒 C3 信息增长裕度深度诊断
- 🔒 多实验对比分析
- 🔒 前向预测（C3 零线穿越预警）
- 🔒 实时 ProbeServer 监控
- 🔒 内置理论验证实验
- 🔒 参数扫描与敏感性分析
- 🔒 自洽测试（拟合器可信度验证）
- 🔒 JSON / HTML 导出

---

## 核心理论：C3 判据

训练机器学习模型本质上是一个**能量驱动的信息创生过程**。五环耗散系统理论从热力学第一性原理出发，推导出信息持续增长（$\dot{m} > 0$）的充要条件——**C3 判据**：

$$-\nabla \cdot \mathbf{w} > T(f - \nabla \cdot \mathbf{J}_s)$$

简言之：**信息增长需要能量汇聚与熵排出同时满足**。一个接受充足算力但缺乏"熵排出通道"的训练过程——例如学习率不当、数据质量差——无论 GPU 跑多久，都无法真正学到东西。

C3 裕度量化了"还能学多久"，它通常在 loss 曲线变平**之前**就已经开始下降，因此可以作为早期预警信号。

> 📄 理论全文：[Zenodo 10.5281/zenodo.20533606](https://doi.org/10.5281/zenodo.20533606) (CC BY 4.0)

| 诊断适用性 | 示例 | 适用？ |
|-----------|------|:--:|
| 玩具级 | MNIST、Fashion-MNIST | 不适用 |
| 入门级 | CIFAR-10 + ResNet-18 | **最低门槛** |
| 标准级 | ImageNet、COCO、WikiText-2 | 适用 |
| 工业级 | LLM 预训练/SFT、推荐系统 | 最佳对象 |

---

## Diagnosis Product / 诊断产品

五环诊断器将上述理论框架应用于 ML 训练日志的因果诊断。**有经验的 ML 工程师通过观察 loss/accuracy 曲线也能得出类似的优化建议，但诊断器的价值不在于建议本身，而在于：**

- **量化**：C3 裕度从 10.19 降到 3.90（Δ = -6.29），而非"曲线好像变平了"
- **可复现**：固定随机种子 + Latin Hypercube 采样，同一份 CSV 每次诊断结果完全一致
- **不依赖经验**：诊断判据来自热力学第一性原理推导出的 C3 充要条件，而非个人训练直觉
- **可作为证据**：客户可以拿诊断报告给团队/老板看——"C3 裕度在 epoch 10 后下降了 62%，系统建议降学习率"

---

## Demos & Case Studies / 演示案例

> 所有数据均为公开来源，零私有信息。诊断器从标准 loss/acc 中提取了传统方法看不到的信息。

| 入口 | 链接 | 说明 |
|------|------|------|
| **W&B 公开 Report** | [五环诊断横向对比](https://wandb.ai/baiyaoseshi-self-employed/pentacyclic-diagnostics/reports/%E4%BA%94%E7%8E%AF%E8%AF%8A%E6%96%AD%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94-%E6%9C%89%E8%AF%84%E4%BC%B0-vs-%E6%97%A0%E8%AF%84%E4%BC%B0--VmlldzoxNzE0NjY1OA) | 完整诊断对比（含数据来源、表格、结论） |
| **有诊断 vs 无诊断** | [demo/五环诊断横向对比_有诊断vs无诊断.md](demo/五环诊断横向对比_有诊断vs无诊断.md) | 200 epoch vs 11 epoch，诊断器如何区分 |
| **ResNet 深度分析** | [demo/五环诊断横向对比_ResNet深度_CIFAR10.md](demo/五环诊断横向对比_ResNet深度_CIFAR10.md) | ResNet-20/32/44/56 五环动力学差异 |
| **演示 CSV** | [demo/demo_cifar10_small.csv](demo/demo_cifar10_small.csv) / [demo/demo_reject.csv](demo/demo_reject.csv) | 上传体验：正常诊断 vs 拒诊 |
| **原始数据** | [demo/wandb_logs/](demo/wandb_logs/) / [demo/wandb_logs_raw/](demo/wandb_logs_raw/) | 下载 CSV 自己跑诊断 |

---

## Project Structure / 项目结构

```
pentacyclic-community/
├── app.py              # Streamlit Web 应用
├── requirements.txt    # Python 依赖
├── pyproject.toml      # 项目元数据
├── LICENSE             # Apache 2.0
├── README.md           # 本文件
└── demo/               # 演示案例与数据（详见 demo/README.md）
    ├── 五环诊断横向对比_*.md         # 诊断报告
    ├── demo_*.csv                   # 演示用 CSV
    ├── fetch_wandb_logs.py 等       # W&B 数据拉取工具
    ├── wandb_logs/                  # 有诊断组原始数据
    └── wandb_logs_raw/              # 无诊断组原始数据
```

---

## References / 参考文献

The core theory paper / 核心理论论文：
- [Zenodo 10.5281/zenodo.20533606](https://doi.org/10.5281/zenodo.20533606) (CC BY 4.0)

Key foundational works / 关键基础文献：

1. Prigogine, I. (1967). *Introduction to Thermodynamics of Irreversible Processes*. Wiley.
2. Schrödinger, E. (1944). *What is Life?* Cambridge University Press.
3. Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM J. Res. Dev.*, 5, 183–191.
4. Brillouin, L. (1953). The negentropy principle of information. *J. Appl. Phys.*, 24, 1152–1163.
5. Carnot, S. (1824). *Réflexions sur la puissance motrice du feu*. Bachelier.
6. Shannon, C. E. (1948). A mathematical theory of communication. *Bell Syst. Tech. J.*, 27, 379–423.
7. de Groot, S. R. & Mazur, P. (1962). *Non-Equilibrium Thermodynamics*. North-Holland.
8. Kotas, T. J. (1985). *The Exergy Method of Thermal Plant Analysis*. Butterworths.

---

## License / 许可

本仓库代码采用 [Apache License 2.0](LICENSE) 许可。

理论论文采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。

---

## Author / 作者

**唐昊东 (Tang Haodong)** — 五环耗散系统理论创立者

- 📧 937692907@qq.com
- 🏠 长沙，中国
