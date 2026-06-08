"""拉取 hhoanguet/Lightning-Cifar10 数据，按 epoch 聚合并转换为 PentacyclicSim 格式。"""
import wandb
import csv
import os
import pandas as pd
import numpy as np

api = wandb.Api()

entity = "hhoanguet"
project = "Lightning-Cifar10"
output_dir = "demo/wandb_logs"

print(f"拉取 {entity}/{project} ...")
runs = list(api.runs(f"{entity}/{project}"))
print(f"  找到 {len(runs)} 条 run")

os.makedirs(output_dir, exist_ok=True)

for run in runs:
    print(f"\n{'='*50}")
    print(f"  [{run.name}] state={run.state}")
    
    df = run.history(samples=5000)
    if df.empty:
        print(f"    无数据，跳过")
        continue
    
    cols = list(df.columns)
    print(f"    原始列: {cols}")
    print(f"    原始行数: {len(df)}")
    
    # 找到 epoch 列
    epoch_col = None
    for c in cols:
        if c == 'epoch' or c == 'epochs':
            epoch_col = c
            break
    
    if epoch_col is None:
        print(f"    无 epoch 列，跳过")
        continue
    
    # 去掉 NaN epoch 的行
    df = df.dropna(subset=[epoch_col])
    df[epoch_col] = df[epoch_col].astype(int)
    
    # 映射列名
    train_loss_col = None
    val_loss_col = None
    val_acc_col = None
    
    for c in cols:
        cl = c.lower()
        if 'train' in cl and 'loss' in cl:
            train_loss_col = c
        if ('val' in cl or 'test' in cl) and 'loss' in cl:
            val_loss_col = c
        if ('val' in cl or 'test' in cl) and ('acc' in cl or 'error' in cl or 'classif' in cl):
            val_acc_col = c
    
    print(f"    列映射: epoch={epoch_col}, train_loss={train_loss_col}, val_loss={val_loss_col}, val_acc={val_acc_col}")
    
    if not train_loss_col:
        print(f"    缺少 train_loss，跳过")
        continue
    
    # 按 epoch 聚合
    epochs = sorted(df[epoch_col].unique())
    print(f"    epoch 数: {len(epochs)}, 范围: {min(epochs)}-{max(epochs)}")
    
    csv_path = os.path.join(output_dir, f"{run.name}.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        writer.writerow(header)
        print(f"    CSV 头: {header}")
        
        for ep in epochs:
            ep_data = df[df[epoch_col] == ep]
            
            t_loss = ep_data[train_loss_col].mean()
            t_acc = 0  # 该项目无 train_acc
            
            v_loss = 0
            v_acc = 0
            
            if val_loss_col and val_loss_col in ep_data.columns:
                v_loss_vals = ep_data[val_loss_col].dropna()
                if len(v_loss_vals) > 0:
                    v_loss = v_loss_vals.mean()
            
            if val_acc_col and val_acc_col in ep_data.columns:
                v_acc_vals = ep_data[val_acc_col].dropna()
                if len(v_acc_vals) > 0:
                    raw = v_acc_vals.mean()
                    # 如果是 error rate，转为 accuracy
                    if 'error' in val_acc_col.lower():
                        v_acc = 1.0 - raw
                    else:
                        v_acc = raw
            
            writer.writerow([ep, t_loss, t_acc, v_loss, v_acc])
    
    print(f"    已导出 -> {csv_path} ({len(epochs)} epochs)")

print(f"\n完成！文件在 {output_dir}/")
