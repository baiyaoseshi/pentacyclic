# Demo & Case Studies / 演示案例

> **这不是一个可在线运行的 Demo。** 这是一个**案例展示区**——我们在此展示五环诊断器在公开训练日志上的真实诊断结果。
> 
> 如果你想自己跑诊断：上传 `demo_cifar10_small.csv` 或 `demo_reject.csv` 到社区版 Streamlit 应用（本仓库 `app.py`），即可体验正常诊断和拒诊两种模式。

---

## 如何阅读这些材料

### 第 1 步：看 W&B 公开 Report（5 分钟）

打开 👉 [五环诊断横向对比：有评估 vs 无评估](https://wandb.ai/baiyaoseshi-self-employed/pentacyclic-diagnostics/reports/%E4%BA%94%E7%8E%AF%E8%AF%8A%E6%96%AD%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94-%E6%9C%89%E8%AF%84%E4%BC%B0-vs-%E6%97%A0%E8%AF%84%E4%BC%B0--VmlldzoxNzE0NjY1OA)

这份 W&B Report 有完整的排版、表格和结论，是我们推荐的**首选入口**。不需要了解技术细节，看完就能理解诊断器在做什么。

### 第 2 步：看详细分析报告（10 分钟）

本目录下的两份 Markdown 报告：

| 文件 | 内容 |
|------|------|
| [五环诊断横向对比_有诊断vs无诊断.md](./五环诊断横向对比_有诊断vs无诊断.md) | 两组公开数据的诊断对比：为什么"有诊断"的数据能出有意义的结论，而"无诊断"的数据诊断器直接拒诊 |
| [五环诊断横向对比_ResNet深度_CIFAR10.md](./五环诊断横向对比_ResNet深度_CIFAR10.md) | 4 种不同深度的 ResNet 在 CIFAR-10 上的五环动力学差异分析 |

### 第 3 步：看原始数据（可选）

[wandb_logs/](./wandb_logs/) 和 [wandb_logs_raw/](./wandb_logs_raw/) 里的 CSV 文件是诊断的原始输入。每一行是训练过程中一个 epoch 的 loss/accuracy。

这些数据全部来自 W&B 公开项目，你可以：
- 用 Excel 打开看原始曲线
- 上传到社区版 Streamlit 应用（本仓库 `app.py`）自己跑诊断

---

## 报告中关键术语解释

| 术语 | 通俗解释 |
|------|----------|
| **联合 R²** | 诊断器对训练过程建模的拟合质量。>0.6 说明模型有效描述了训练动力学 |
| **C3 裕度** | "信息还能继续增长多久"的量化指标。越高越好，下降说明学习在放缓 |
| **g** (储存能量) | 模型已固化的"知识"。loss 越低、accuracy 越高 → g 越高 |
| **e** (㶲/可用能) | 训练还剩多少"燃料"。从 100% 开始，随训练消耗 |
| **g-only 模式** | 输入数据不完整时诊断器的安全模式——宁可少给结论，不给错结论 |
| **e 变幅** | e 在整个训练过程中的变化幅度。越大说明训练动力学越丰富 |

---

## 常见问题

**Q: 这里的东西能直接运行吗？**
A: 不能。这个目录展示的是诊断**结果**和**原始数据**。诊断引擎是独立的 C#/.NET 程序（见仓库 [README](../README.md#快速开始)）。但你可以用本仓库的社区版 Streamlit 应用（`app.py`）上传本目录的 CSV 自行诊断。

**Q: 你们的诊断器和 ML 工程师的经验判断有什么区别？**
A: 有经验的工程师看 loss/accuracy 曲线也能发现"后期该降学习率"。我们的区别在于**量化**：不是"曲线变平了"，而是"C3 裕度从 2.53 降到 1.56，降幅 38%"。量化的结论可以写进报告、作为决策依据、给老板看。

**Q: 诊断器需要我的私有数据吗？**
A: 不需要。诊断器只需要标准训练日志（epoch, train_loss, val_loss, val_acc）。你不需要提供模型权重、数据集或代码。

**Q: 怎么试用？**
A: 按照仓库 [README](../README.md#快速开始) 启动社区版 Streamlit 应用，上传本目录的 `demo_cifar10_small.csv` 或 `demo_reject.csv` 即可体验。如需完整诊断功能（6 参数拟合、C3 裕度详细分析等），请联系作者。

---

## 数据来源声明

本目录中所有训练日志 CSV 均来自 W&B 公开项目：
- [hhoanguet/Lightning-Cifar10](https://wandb.ai/hhoanguet/Lightning-Cifar10)（4 组 ResNet 架构对比）
- [wandb/wandb-lightning](https://wandb.ai/wandb/wandb-lightning)（官方 PyTorch Lightning 教程）

未使用任何私有数据，未泄露任何第三方信息。
