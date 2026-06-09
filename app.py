"""
五环耗散系统 · 训练诊断工具 社区版
Pentacyclic Diagnostics Community Edition

基于热力学-信息论的 ML 训练因果诊断引擎。
上传训练日志 CSV → ODE 参数拟合 → C3 信息增长裕度评估 → 优化建议。

开源许可: Apache License 2.0
理论论文: https://doi.org/10.5281/zenodo.20533606 (CC BY 4.0)

依赖: PentacyclicSim C# 后端 (https://github.com/tanghaodong/pentacyclic-sim)
"""

import streamlit as st
import pandas as pd
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import subprocess
import tempfile
import re
import os
from pathlib import Path

# ═══════════════════════════════════════════
#  全局配置
# ═══════════════════════════════════════════
# PentacyclicSim 后端路径：支持环境变量、自动检测、默认搜索
_PENTA_DIR_ENV = os.environ.get("PENTACYCLIC_SIM_DIR", "")
if _PENTA_DIR_ENV:
    PENTA_DIR = Path(_PENTA_DIR_ENV)
else:
    # 自动搜索：先找同级目录，再找当前目录
    _candidates = [
        Path(__file__).resolve().parent.parent / "PentacyclicSim",
        Path(__file__).resolve().parent.parent / "数值模拟" / "PentacyclicSim",
        Path.cwd() / "PentacyclicSim",
    ]
    PENTA_DIR = next((c for c in _candidates if (c / "PentacyclicSim.csproj").exists()), Path.cwd())

# EXE 查找（Release > Debug）
_EXE_CANDIDATES = ["net10.0", "net9.0", "net8.0"]
EXE_PATH = None
for _cfg in ("Release", "Debug"):
    for _tf in _EXE_CANDIDATES:
        _p = PENTA_DIR / "bin" / _cfg / _tf / "PentacyclicSim.exe"
        if _p.exists():
            EXE_PATH = _p
            break
    if EXE_PATH:
        break

RESULTS_DIR = PENTA_DIR / "results"
DIAGNOSIS_TIMEOUT = 1800  # 秒

# 已知 g 列关键字（按优先级）
G_COLUMN_KEYWORDS = [
    "train_acc", "train_accuracy", "accuracy", "acc",
    "train_loss", "train/loss", "loss", "training_loss",
]
# 已知 e 列关键字
E_COLUMN_KEYWORDS = [
    "val_loss", "val/loss", "validation_loss",
    "val_acc", "val_accuracy", "validation_acc", "validation_accuracy",
]

# ═══════════════════════════════════════════
#  页面设置
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="五环诊断 社区版",
    page_icon="🔬",
    layout="wide",
)

# 隐藏 Streamlit 默认 UI
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🔬 五环耗散系统 · 训练诊断 社区版")
st.markdown(
    "上传您的训练日志 CSV，系统将自动拟合耗散动力学参数，"
    "评估 C3 信息增长裕度，并给出资源瓶颈与优化建议。"
    "诊断过程需要 5-10 分钟，请耐心等待。"
)

# ═══════════════════════════════════════════
#  能力范围说明
# ═══════════════════════════════════════════
with st.expander("📋 诊断能力与适用范围（请先阅读）"):
    st.markdown("""
### 适用场景
- **监督学习训练任务**（分类、回归）：含 epoch 级 `train_loss` / `train_acc` / `val_loss` / `val_acc` 日志
- **典型规模**：CIFAR-10 / CIFAR-100 / ImageNet 等标准 benchmark，模型从小型 CNN 到 ResNet-18 以上
- **数据要求**：建议至少 **30 个 epoch** 以上的完整训练日志
- **最低门槛**：CIFAR-10 + ResNet-18（MNIST 级玩具模型不在理论适用范围）

### 诊断的定位
有经验的 ML 工程师通过观察 loss/accuracy 曲线，也能得出类似的优化建议（如"后期该降学习率"）。本产品的差异化不在于建议本身，而在于：
- **量化**：不是"曲线变平了"，而是"信息增长裕度下降了 62%"
- **可复现**：同一份数据每次诊断结果完全一致，不依赖个人状态
- **有理论依据**：判据来自热力学第一性原理推导，可作为团队决策的证据

### 诊断可能失效的情况
| 情况 | 表现 | 原因 |
|------|------|------|
| 观测点 < 20 epoch | R² 偏低或为负 | 信息不足，无法稳定拟合 6 个自由参数 |
| g/e 高度共线（相关系数 > 0.99） | 自动切换 g-only 模式 | 可用能曲线与储存能量曲线几乎同步 |
| g_max / e_max 设置不当 | C3 裕度被高估或低估 | 归一化参考系错误，需调整高级选项 |
| 非 epoch 级日志（如 batch 级） | 拟合失败或参数异常 | 数据密度与 ODE 积分步长不匹配 |
| 非监督学习任务 | 模型不适用 | 理论当前仅覆盖监督学习的能量-信息耦合框架 |

### 社区版限制
- 仅显示 2 项核心拟合参数（Epsilon0, Alpha）
- 前 1 条诊断建议
- 无多实验对比、前向预测、实时监控功能
- 升级 **专业版** 解锁全部功能：[联系我们](mailto:937692907@qq.com)

### 免责声明
本工具为研究原型，诊断结果供参考，不构成生产决策依据。
""")

# ═══════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════


def detect_columns(headers: list[str]):
    """自动检测 g 列和 e 列索引。返回 (g_idx, g_name, e_idx, e_name) 或 (None,)*4"""
    headers_lower = [(i, h.strip().lower()) for i, h in enumerate(headers)]

    g_idx, g_name = None, None
    for kw in G_COLUMN_KEYWORDS:
        for i, h in headers_lower:
            if h == kw:
                g_idx, g_name = i, headers[i].strip()
                break
        if g_idx is not None:
            break

    if g_idx is None:
        for i, h in headers_lower:
            if "train_acc" in h or "train_accuracy" in h:
                g_idx, g_name = i, headers[i].strip()
                break
    if g_idx is None:
        for i, h in headers_lower:
            if "train_loss" in h or "train/loss" in h or "train loss" in h:
                g_idx, g_name = i, headers[i].strip()
                break

    e_idx, e_name = None, None
    for kw in E_COLUMN_KEYWORDS:
        for i, h in headers_lower:
            if h == kw:
                e_idx, e_name = i, headers[i].strip()
                break
        if e_idx is not None:
            break

    if e_idx is None:
        for i, h in headers_lower:
            if "val_acc" in h or "validation_acc" in h:
                e_idx, e_name = i, headers[i].strip()
                break
    if e_idx is None:
        for i, h in headers_lower:
            if "val_loss" in h or "validation_loss" in h or "val/loss" in h:
                e_idx, e_name = i, headers[i].strip()
                break

    return g_idx, g_name, e_idx, e_name


def run_diagnosis(csv_path: str, g_max: float = 100.0, e_max: float = 100.0) -> dict:
    """直接执行编译好的 PentacyclicSim.exe"""
    if EXE_PATH is None:
        return {
            "stdout": "",
            "stderr": "PentacyclicSim.exe 未找到。请先编译 C# 后端:\n"
                      f"  cd {PENTA_DIR}\n  dotnet build -c Release",
            "returncode": 1,
        }

    subprocess.run(
        ["taskkill", "/f", "/im", "PentacyclicSim.exe"],
        capture_output=True, timeout=5,
    )
    cmd = [
        str(EXE_PATH),
        "--full-report", csv_path,
        str(g_max), str(e_max),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(PENTA_DIR),
        capture_output=True,
        text=True,
        timeout=DIAGNOSIS_TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def parse_params_from_stdout(stdout: str) -> dict:
    """从 stdout 中提取拟合参数和 R²"""
    params = {}
    param_patterns = {
        "Epsilon0": r"Epsilon0\s*=\s*([\d\.\-eE\+]+)",
        "EtaRegen": r"EtaRegen\s*=\s*([\d\.\-eE\+]+)",
        "Alpha": r"Alpha\s*=\s*([\d\.\-eE\+]+)",
        "Sigma": r"Sigma\s*=\s*([\d\.\-eE\+]+)",
        "EntropyDissipation": r"EntropyDissipation\s*=\s*([\d\.\-eE\+]+)",
        "ForgetRate": r"ForgetRate\s*=\s*([\d\.\-eE\+]+)",
    }
    for name, pat in param_patterns.items():
        m = re.search(pat, stdout)
        if m:
            params[name] = float(m.group(1))

    m = re.search(r"g-R2\s*=\s*([\d\.\-eE]+)\s*e-R2\s*=\s*([\d\.\-eE]+)\s*联合\s*=\s*([\d\.\-eE]+)", stdout)
    if m:
        params["g_R2"] = float(m.group(1))
        params["e_R2"] = float(m.group(2))
        params["joint_R2"] = float(m.group(3))

    g_only_patterns = [r"仅拟合\s*g", r"g-only", r"e\s*为占位值"]
    params["g_only"] = any(re.search(p, stdout, re.IGNORECASE) for p in g_only_patterns)

    return params


def compute_safe_name(csv_stem: str) -> str:
    """与 C# RunFullReport 的 safeName 逻辑一致"""
    stem = csv_stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"全流程_{stem}"


def read_fitted_csv(safe_name: str) -> pd.DataFrame | None:
    fitted_path = RESULTS_DIR / f"{safe_name}_fitted.csv"
    if fitted_path.exists():
        return pd.read_csv(fitted_path)
    return None


def extract_suggestions(safe_name: str) -> list[str]:
    """从 C# 后端生成的 full_report_*.md 中提取 Section 5 诊断建议"""
    report_path = RESULTS_DIR / f"full_report_{safe_name}.md"
    if not report_path.exists():
        return []

    suggestions = []
    in_section = False
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            if "## 5. 优化建议列表" in line:
                in_section = True
                continue
            if in_section:
                if line.startswith("## "):
                    break
                stripped = line.strip()
                if stripped.startswith("1. ") or stripped.startswith("- "):
                    clean = re.sub(r"^\d+\.\s*", "", stripped)
                    clean = clean.replace("**", "")
                    if clean and "✅" not in clean:
                        suggestions.append(clean)
    return suggestions


# ═══════════════════════════════════════════
#  文件上传
# ═══════════════════════════════════════════
st.header("📂 上传训练日志 CSV")
uploaded_file = st.file_uploader(
    "拖拽或点击上传 CSV 文件",
    type=["csv"],
    help="支持标准列名: epoch, train_loss, train_acc, val_loss, val_acc（及变体）",
)

if uploaded_file is not None:
    df_orig = pd.read_csv(uploaded_file)
    headers = list(df_orig.columns)

    # ── 后端状态检测 ──
    if EXE_PATH is None:
        st.warning(f"""
        ⚠️ **PentacyclicSim 后端未编译**
        
        社区版需要 C# 诊断引擎后端。请先编译:
        ```bash
        cd {PENTA_DIR}
        dotnet build -c Release
        ```
        或设置环境变量 `PENTACYCLIC_SIM_DIR` 指向项目目录。
        """)

    st.caption(f"已加载 {len(df_orig)} 行 × {len(headers)} 列")

    # ── 列映射 ──
    g_idx, g_name, e_idx, e_name = detect_columns(headers)

    col1, col2 = st.columns(2)
    has_auto_g = g_idx is not None
    has_auto_e = e_idx is not None
    all_auto = has_auto_g and has_auto_e

    if all_auto:
        with col1:
            st.success(f"✅ g 列（储存能量）自动匹配: **{g_name}**（第 {g_idx + 1} 列）")
        with col2:
            st.success(f"✅ e 列（可用学习空间）自动匹配: **{e_name}**（第 {e_idx + 1} 列）")
    else:
        st.warning("⚠ 部分列名未自动识别，请手动选择：")
        col3, col4 = st.columns(2)
        with col3:
            g_sel = st.selectbox(
                "g 列（储存能量，如 train_acc）",
                headers,
                index=g_idx if g_idx is not None else 0,
            )
            g_idx = headers.index(g_sel)
            g_name = g_sel
        with col4:
            e_sel = st.selectbox(
                "e 列（可用学习空间，如 val_acc）",
                headers,
                index=e_idx if e_idx is not None else 0,
            )
            e_idx = headers.index(e_sel)
            e_name = e_sel

    # ── 高级选项 ──
    with st.expander("⚙ 高级选项（系统尺度锚点）"):
        st.caption(
            "这两个值定义了训练系统的**归一化参考系**，诊断器依赖它们判断增长饱和与资源充裕度。"
            "默认值 100 适用于 CIFAR-10 / ResNet 级别。"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            g_max = st.number_input(
                "g_max — 存储容量上限",
                value=100.0, min_value=1.0, max_value=100000.0,
                help="模型结构理论上能承载的信息上限。由参数量、架构容量、任务复杂度决定。"
                     "设太高会低估增长，设太低会误判饱和。"
                     "建议：小型任务 50–200，中型任务 200–2000，大模型 10³–10⁶。"
            )
        with col_b:
            e_max = st.number_input(
                "e_max — 数据充裕度上限",
                value=100.0, min_value=1.0, max_value=100000.0,
                help="数据集所能提供的最大可用学习空间。由数据量、标签质量、信息密度决定。"
                     "设太低会高估 C3 裕度，设太高会低估资源利用率。"
                     "建议与 g_max 同量级。"
            )

    # ── 开始诊断按钮 ──
    st.divider()
    if st.button("🚀 开始诊断", type="primary", use_container_width=True):
        if EXE_PATH is None:
            st.error("❌ 诊断引擎未编译。请在 PentacyclicSim 目录执行 `dotnet build -c Release`。")
            st.stop()

        # 使用原始文件名创建临时文件（确保 Python 和 C# 后端用同一 safeName）
        temp_dir = tempfile.mkdtemp()
        tmp_csv_path = os.path.join(temp_dir, uploaded_file.name)
        df_orig.to_csv(tmp_csv_path, index=False)

        safe_name = compute_safe_name(Path(uploaded_file.name).stem)

        try:
            with st.status("正在运行五环诊断管道...", expanded=True) as status:
                status.update(label="正在调用诊断引擎（ODE 参数拟合，约 5-10 分钟）...")
                diag = run_diagnosis(tmp_csv_path, g_max, e_max)

                if diag["returncode"] != 0:
                    st.error(f"❌ 诊断引擎返回错误码 {diag['returncode']}")
                    if diag["stderr"]:
                        st.code(diag["stderr"], language="text")
                    st.stop()

                stdout = diag["stdout"]
                status.update(label="诊断完成，正在解析结果...")
                params = parse_params_from_stdout(stdout)
                fitted_df = read_fitted_csv(safe_name)
                status.update(label="诊断完成 ✅", state="complete")

            # ═══════════════════════════════════
            #  结果展示
            # ═══════════════════════════════════
            st.header("📊 诊断报告")

            g_only = params.get("g_only", False)

            # ── g-only 模式警告 ──
            if g_only:
                st.warning(
                    "⚠️ **仅拟合了 g（储存能量），e（可用学习空间）未参与拟合。**\n\n"
                    "原因：e 观测值变化太小（<20%）或与 g 高度共线，无法提供独立约束。"
                    "下方的 e-R² 为占位值，e 拟合曲线不代表真实观测。",
                )

            # ── 核心参数（仅展示 2 个）──
            st.subheader("核心拟合参数")
            core_params = []
            if "Epsilon0" in params:
                core_params.append({"参数": "Epsilon0（学习效率）", "拟合值": f"{params['Epsilon0']:.4f}"})
            if "Alpha" in params:
                core_params.append({"参数": "Alpha（数据质量系数）", "拟合值": f"{params['Alpha']:.4f}"})
            if core_params:
                st.dataframe(pd.DataFrame(core_params), use_container_width=True, hide_index=True)
                st.caption("🔒 专业版含 EtaRegen、Sigma、ForgetRate 等全部 6 项参数及物理含义解读")

            # R²
            st.subheader("拟合优度")
            r2_cols = st.columns(3)
            with r2_cols[0]:
                g_r2 = params.get("g_R2", None)
                st.metric("g-R²", f"{g_r2:.4f}" if g_r2 else "—")
            with r2_cols[1]:
                e_r2 = params.get("e_R2", None)
                if g_only:
                    st.metric("e-R²", "N/A", help="g-only 模式：e 未参与拟合，此值为占位符")
                elif e_r2:
                    st.metric("e-R²", f"{e_r2:.4f}")
                else:
                    st.metric("e-R²", "—")
            with r2_cols[2]:
                joint_r2 = params.get("joint_R2", None)
                st.metric("联合 R²", f"{joint_r2:.4f}" if joint_r2 else "—")

            # ── g/e 对比图 ──
            st.subheader("真实值 vs 拟合值对比")
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

            # g 对比
            ax1.set_title("g（储存能量）: 观测 vs 拟合", fontsize=13)
            if g_idx is not None:
                g_raw = pd.to_numeric(df_orig.iloc[:, g_idx], errors="coerce")
                if g_name and "loss" in g_name.lower():
                    g_obs = (1 - g_raw / g_raw.iloc[0]) * g_max
                else:
                    g_obs = g_raw * g_max
                ax1.plot(
                    range(len(g_obs)), g_obs.values,
                    "o", markersize=3, alpha=0.6, label="观测值", color="#1f77b4",
                )
            if fitted_df is not None and "g(结构能量)" in fitted_df.columns:
                ax1.plot(
                    fitted_df["时间"], fitted_df["g(结构能量)"],
                    "-", linewidth=2, label="拟合值", color="#ff7f0e",
                )
                ax1.set_xlabel("时间")
            else:
                ax1.set_xlabel("Epoch")
            ax1.set_ylabel("g")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # e 对比
            e_title = "e（可用学习空间）: 观测 vs 拟合"
            if g_only:
                e_title += " [未参与拟合]"
            ax2.set_title(e_title, fontsize=13)
            if e_idx is not None:
                e_raw = pd.to_numeric(df_orig.iloc[:, e_idx], errors="coerce")
                if e_name and "loss" in e_name.lower():
                    e_obs = e_raw / e_raw.iloc[0] * e_max
                else:
                    e_obs = e_raw * e_max
                ax2.plot(
                    range(len(e_obs)), e_obs.values,
                    "o", markersize=3, alpha=0.6, label="观测值", color="#1f77b4",
                )
            if fitted_df is not None and "e(数据资源)" in fitted_df.columns:
                ax2.plot(
                    fitted_df["时间"], fitted_df["e(数据资源)"],
                    "-", linewidth=2,
                    label="ODE 自由演化（非拟合）" if g_only else "拟合值",
                    color="#d62728" if g_only else "#2ca02c",
                    linestyle="--" if g_only else "-",
                )
                ax2.set_xlabel("时间")
            else:
                ax2.set_xlabel("Epoch")
            ax2.set_ylabel("e")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig1.tight_layout()
            st.pyplot(fig1)

            # ── C3 裕度曲线 ──
            if fitted_df is not None and "C3裕度" in fitted_df.columns:
                st.subheader("C3 裕度（信息增长条件）")
                fig2, ax3 = plt.subplots(figsize=(12, 3))
                c3 = fitted_df["C3裕度"].values
                t = fitted_df["时间"].values

                # 填充：C3 > 0 绿色，C3 < 0 红色
                ax3.fill_between(t, 0, c3,
                    where=(c3 >= 0), color="#2ca02c", alpha=0.25, label="C3 ≥ 0（正常）")
                ax3.fill_between(t, c3, 0,
                    where=(c3 < 0), color="#d62728", alpha=0.35, label="C3 < 0（违规）")

                ax3.plot(t, c3, "-", linewidth=2, color="#1f77b4")
                ax3.axhline(y=0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7)

                # 统计标注
                avg_c3 = float(c3.mean())
                viol_pct = float((c3 < 0).mean() * 100)
                ax3.set_title(
                    f"C3 裕度（信息增长充要条件）  —  均值 {avg_c3:.4f}  |  违规 {viol_pct:.1f}%",
                    fontsize=13,
                )
                ax3.set_xlabel("时间")
                ax3.set_ylabel("C3 裕度")
                ax3.legend(loc="upper right")
                ax3.grid(True, alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2)

            # ── 建议（社区版：仅第 1 条）──
            st.subheader("💡 诊断建议")
            suggestions = extract_suggestions(safe_name)
            if suggestions:
                for s in suggestions[:1]:
                    st.markdown(f"- {s}")
            else:
                st.info("✅ 当前训练配置未触发预设优化阈值，系统运行良好。")

            # ── 专业版引导 ──
            st.divider()
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 1.5rem 2rem;
                margin: 1rem 0;
                color: white;
                text-align: center;
            ">
                <h3 style="margin: 0 0 0.5rem 0; color: white;">🔒 升级专业版</h3>
                <p style="margin: 0 0 0.3rem 0; opacity: 0.95;">
                    全部 6 项拟合参数 · 五变量演化分析 · C3 裕度深度诊断 · 多实验对比 · 前向预测 · 实时监控
                </p>
                <p style="margin: 0 0 1rem 0; opacity: 0.85;">
                    内置实验验证 · 参数扫描 · 自洽测试 · JSON/HTML 导出
                </p>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.75;">
                    📧 937692907@qq.com
                </p>
            </div>
            """, unsafe_allow_html=True)

        except subprocess.TimeoutExpired:
            st.error(f"❌ 诊断超时（超过 {DIAGNOSIS_TIMEOUT // 60} 分钟）。")
        except Exception as e:
            st.error(f"❌ 诊断过程出错: {e}")
        finally:
            try:
                os.unlink(tmp_csv_path)
                os.rmdir(temp_dir)
            except OSError:
                pass

# ═══════════════════════════════════════════
#  底部
# ═══════════════════════════════════════════
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #888; padding: 1.5rem 0;">

### 想要更多功能？

升级 **专业版** 获取完整诊断能力：
多实验对比 · 前向预测 · 实时监控 · 内置实验验证 · 参数扫描 · 自洽测试

📧 **937692907@qq.com**

---

*Pentacyclic Diagnostics Community Edition — Apache 2.0*
*理论论文: [Zenodo 10.5281/zenodo.20533606](https://doi.org/10.5281/zenodo.20533606) (CC BY 4.0)*

</div>
""",
    unsafe_allow_html=True,
)
