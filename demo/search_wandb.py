"""搜索 W&B 公开训练日志 - 改进版：尝试多个已知路径 + 广泛搜索。"""
import wandb
import sys

api = wandb.Api()

# 尝试已知的公开项目路径
known_paths = [
    "geekyutao/cifar10-quickstart",
    "cifar10",
    "cifar-10",
    "resnet-cifar10",
    "pytorch-cifar",
    "cifar10-baseline",
    "cifar10-classifier",
    "wandb/cifar10-benchmark",
]

# 尝试用 MongoDB 风格查询搜索公开 runs
try:
    # 先用空路径尝试
    print("尝试广泛搜索...")
    runs = list(api.runs(filters={"$and": [{"tags": {"$nin": [""]}}]}))
    print(f"  找到 {len(runs)} runs")
    for r in runs[:5]:
        print(f"  {r.entity}/{r.project}/{r.name}")
except Exception as e:
    print(f"  广泛搜索失败: {e}")

# 逐个尝试已知项目
print("\n--- 尝试已知项目 ---")
for path in known_paths:
    try:
        runs = list(api.runs(path))
        if runs:
            print(f"\n[{path}] 找到 {len(runs)} runs:")
            for r in runs[:5]:
                # 获取历史
                try:
                    hist = r.history(samples=500)
                    cols = list(hist.columns)
                    has_loss = any("loss" in c.lower() for c in cols)
                    has_acc = any("acc" in c.lower() or "accuracy" in c.lower() for c in cols)
                    print(f"  {r.name} [{r.state}] rows={len(hist)} loss={has_loss} acc={has_acc} cols={cols[:10]}")
                except Exception as e2:
                    print(f"  {r.name} [{r.state}] (无法获取历史: {e2})")
    except Exception as e:
        pass

print("\n完成。")
