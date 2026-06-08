"""拉取 wandb/wandb-lightning（无诊断原始日志）并转换。"""
import wandb
import csv
import os

api = wandb.Api()
entity = "wandb"
project = "wandb-lightning"
output_dir = "demo/wandb_logs_raw"

runs = list(api.runs(f"{entity}/{project}"))
print(f"找到 {len(runs)} 条 run")
os.makedirs(output_dir, exist_ok=True)

for run in runs:
    print(f"\n{'='*50}")
    print(f"  [{run.name}] state={run.state}")
    
    df = run.history(samples=5000)
    cols = list(df.columns)
    print(f"  列: {cols}")
    
    # 找 epoch 列
    epoch_col = None
    for c in cols:
        if c == 'epoch' or c == 'epochs':
            epoch_col = c
            break
    
    if epoch_col is None:
        print(f"  无 epoch 列，跳过")
        continue
    
    df = df.dropna(subset=[epoch_col])
    if epoch_col in df.columns:
        df[epoch_col] = df[epoch_col].astype(int)
    
    # 映射
    train_loss_col = None
    val_loss_col = None
    val_acc_col = None
    
    for c in cols:
        cl = c.lower()
        if 'train' in cl and 'loss' in cl:
            train_loss_col = c
        if ('val' in cl or 'test' in cl) and 'loss' in cl:
            val_loss_col = c
        if ('val' in cl or 'test' in cl) and ('acc' in cl or 'error' in cl):
            val_acc_col = c
    
    print(f"  映射: epoch={epoch_col}, train_loss={train_loss_col}, val_loss={val_loss_col}, val_acc={val_acc_col}")
    
    if not train_loss_col:
        print(f"  缺少 train_loss，跳过")
        continue
    
    epochs = sorted(df[epoch_col].unique())
    print(f"  epochs: {min(epochs)}-{max(epochs)} ({len(epochs)}个)")
    
    csv_path = os.path.join(output_dir, f"{run.name.replace('/', '_')}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        
        for ep in epochs:
            ep_data = df[df[epoch_col] == ep]
            t_loss = ep_data[train_loss_col].mean() if train_loss_col in ep_data.columns else 0
            
            v_loss = 0
            if val_loss_col and val_loss_col in ep_data.columns:
                v = ep_data[val_loss_col].dropna()
                if len(v) > 0:
                    v_loss = v.mean()
            
            v_acc = 0
            if val_acc_col and val_acc_col in ep_data.columns:
                v = ep_data[val_acc_col].dropna()
                if len(v) > 0:
                    v_acc = v.mean()
            
            writer.writerow([ep, t_loss, 0, v_loss, v_acc])
    
    print(f"  已导出 -> {csv_path}")

print(f"\n完成！文件在 {output_dir}/")
