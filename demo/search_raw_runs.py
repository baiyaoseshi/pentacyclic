"""搜索「无诊断」的公开训练日志 —— 单次跑、无架构对比、无超参扫描的原始训练。"""
import wandb

api = wandb.Api()

# 候选「无诊断」项目 —— 教程 demo、个人随手跑
candidates = [
    # 官方教程 demo（无分析）
    "wandb/wandb-lightning",
    "wandb/pytorch-demo", 
    # 个人随手跑
    "flora-ufsc24/Test-Metrics",
    "caydenwei/cifar10-pytorch",
    # 更多尝试
    "mostafaibrahim17/ml-articles",
    "geekyutao/cifar10-quickstart",
    "ayulockin/wandb-tutorial",
    "capecape/wandb-tutorial",
]

good_runs = []

for path in candidates:
    try:
        runs = list(api.runs(path))
        if runs:
            for r in runs[:5]:
                # 获取历史看看有没有 epoch 级 loss/acc
                try:
                    hist = r.history(samples=5000)
                    cols = [c.lower() for c in hist.columns]
                    has_epoch = any('epoch' in c for c in cols)
                    has_loss = any('loss' in c for c in cols)
                    has_acc = any('acc' in c for c in cols) or any('accuracy' in c for c in cols)
                    
                    if has_loss and has_acc and has_epoch:
                        epochs = hist['epoch'].dropna().unique() if 'epoch' in hist.columns else []
                        good_runs.append({
                            'path': path,
                            'name': r.name,
                            'state': r.state,
                            'rows': len(hist),
                            'epochs': len(epochs),
                            'cols': cols[:8],
                            'tags': ','.join(r.tags[:3]) if r.tags else '无',
                        })
                        print(f"[{path}] {r.name} | {r.state} | rows={len(hist)} | epochs={len(epochs)} | tags={','.join(r.tags[:3]) if r.tags else '无'}")
                except Exception as e:
                    print(f"  [{path}] {r.name}: 读取历史失败 ({e})")
    except Exception as e:
        pass

print(f"\n找到 {len(good_runs)} 个可用的「无诊断」run:")
for g in good_runs:
    print(f"  {g['path']}/{g['name']} | epochs={g['epochs']} | tags={g['tags']}")
