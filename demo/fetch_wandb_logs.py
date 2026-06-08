"""从 W&B 公开项目中拉取训练日志 CSV。

用法：
    python fetch_wandb_logs.py                      # 交互式搜索
    python fetch_wandb_logs.py --entity 用户名 --project 项目名   # 直接拉
    python fetch_wandb_logs.py --query "cifar10 resnet"           # 搜索关键词

输出：
    每条 run 一个 CSV 文件，包含 epoch/train_loss/val_acc 等列。
"""

import argparse
import csv
import os
import sys
import wandb


def search_and_list(api, query, entity=None):
    """搜索公开 runs 并打印列表。"""
    filters = {}
    if entity:
        filters["entity"] = entity

    print(f"搜索: '{query}' ...")
    runs = list(api.runs(filters=filters, per_page=50))

    if not runs:
        print("没找到结果。试试换关键词。")
        return []

    # 按项目分组展示
    projects = {}
    for run in runs:
        tags = ",".join(run.tags) if run.tags else "无"
        key = f"{run.entity}/{run.project}"
        projects.setdefault(key, []).append((run.name, tags, run.state))

    print(f"\n找到 {len(runs)} 条 run，分布在 {len(projects)} 个项目:\n")
    for proj, run_list in projects.items():
        print(f"  [{proj}] ({len(run_list)} runs)")
        for name, tags, state in run_list[:5]:
            print(f"    - {name} [{state}] tags: {tags}")
        if len(run_list) > 5:
            print(f"    ... 等 {len(run_list)-5} 条")
        print()

    return runs


def download_runs(runs, output_dir="wandb_logs"):
    """下载 runs 的 history 为 CSV。"""
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for run in runs:
        try:
            df = run.history()
            if df.empty:
                print(f"  跳过 {run.name}: 无数据")
                continue

            safe_name = run.name.replace("/", "_").replace("\\", "_")
            filename = f"{run.entity}_{run.project}_{safe_name}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False, encoding="utf-8")
            print(f"  已保存 [{run.entity}/{run.project}] {run.name} -> {filepath} ({len(df)} 行)")
            count += 1
        except Exception as e:
            print(f"  下载失败 {run.name}: {e}")

    return count


def main():
    parser = argparse.ArgumentParser(description="从 W&B 拉取公开训练日志")
    parser.add_argument("--entity", "-e", help="W&B 用户名或组织名")
    parser.add_argument("--project", "-p", help="项目名")
    parser.add_argument("--query", "-q", default="cifar10", help="搜索关键词 (默认: cifar10)")
    parser.add_argument("--max-runs", "-n", type=int, default=10, help="最多下载条数 (默认: 10)")
    parser.add_argument("--output", "-o", default="wandb_logs", help="输出目录 (默认: wandb_logs)")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式：先搜后选")
    args = parser.parse_args()

    api = wandb.Api()

    # 模式 1：指定 entity + project，直接拉
    if args.entity and args.project:
        print(f"拉取 {args.entity}/{args.project} ...")
        runs = list(api.runs(f"{args.entity}/{args.project}"))
        if not runs:
            print("该项目无公开 run。")
            return
        print(f"  找到 {len(runs)} 条 run")
        runs = runs[:args.max_runs]
        n = download_runs(runs, args.output)
        print(f"\n完成: 下载 {n}/{len(runs)} 条 run 到 {args.output}/")

    # 模式 2：搜索
    else:
        runs = search_and_list(api, args.query, entity=args.entity)
        if not runs:
            return

        if args.interactive:
            print("输入要下载的项目 (格式: entity/project) 或输入 'all' 全部下载:")
            choice = input("> ").strip()
            if choice.lower() != "all":
                runs = [r for r in runs if f"{r.entity}/{r.project}" == choice]
            if not runs:
                print("无匹配，退出。")
                return

        runs = runs[:args.max_runs]
        n = download_runs(runs, args.output)
        print(f"\n完成: 下载 {n}/{len(runs)} 条 run 到 {args.output}/")

    print("\n提示: 拿这些 CSV 去跑 PentacyclicSim:")
    print("  dotnet run --full-report <csv_path>")


if __name__ == "__main__":
    main()
