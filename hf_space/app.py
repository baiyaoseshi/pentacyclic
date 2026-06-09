"""
五环诊断器 · HuggingFace Space 轻量版
纯 Python 实现，零 C# 依赖。上传训练日志 CSV → ODE 拟合 → C3 裕度诊断。
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import random
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

st.set_page_config(page_title="五环诊断", page_icon="🔬", layout="wide")

# ═══════════════════════════════════════════
#  五环 ODE 核心（内嵌，零依赖）
# ═══════════════════════════════════════════

@dataclass
class Params:
    epsilon0: float = 0.5; alpha: float = 0.8; sigma: float = 0.3
    eta_ex: float = 0.55; delta_forget: float = 0.02; delta_f: float = 0.05
    delta_entropy: float = 0.01; T0: float = 0.2
    g_max: float = 100.0; e_max: float = 100.0
    g0: float = 5.0; e0: float = 80.0; m0: float = 1.0; f0: float = 0.1
    t_end: float = 50.0; dt: float = 0.5
    topology: str = "barrier"

    def effective_epsilon0(self, e: float, g: float) -> float:
        return self.epsilon0 * max(0.0, 1.0 - g / self.g_max)

    def with_overrides(self, **kw):
        d = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        d.update(kw)
        return Params(**d)

@dataclass
class State:
    g: float; e: float; m: float; f: float

def pentacyclic_ode(t: float, state: tuple, p: Params) -> tuple:
    g, e, m, f = state
    # C2: T0 饱和度修正
    T0_correction = 1.0 / (1.0 + p.T0 * f)
    dg = p.epsilon0 * e * max(0.0, 1.0 - g / p.g_max) * T0_correction - p.delta_forget * g
    # e 平衡
    e_regeneration = p.eta_ex * f
    e_consumption = p.epsilon0 * e * max(0.0, 1.0 - g / p.g_max)
    de = e_regeneration - e_consumption - p.delta_entropy * m
    # F2: m 增长
    dm = p.alpha * abs(dg) - p.delta_entropy * m
    # F3: f 来自信息处理
    df = p.sigma * abs(dm) - p.delta_f * f
    return (dg, de, dm, df)

def solve_ode(p: Params, t_end: float = None, dt: float = None) -> List[State]:
    """RK4 求解，返回状态轨迹"""
    t_end = t_end or p.t_end; dt = dt or p.dt
    s = State(g=p.g0, e=p.e0, m=p.m0, f=p.f0)
    states = [State(g=s.g, e=s.e, m=s.m, f=s.f)]
    t = 0.0; steps = int(t_end / dt)
    for _ in range(steps):
        k1 = pentacyclic_ode(t, (s.g, s.e, s.m, s.f), p)
        k2 = pentacyclic_ode(t + dt/2, _add((s.g, s.e, s.m, s.f), _scale(k1, dt/2)), p)
        k3 = pentacyclic_ode(t + dt/2, _add((s.g, s.e, s.m, s.f), _scale(k2, dt/2)), p)
        k4 = pentacyclic_ode(t + dt, _add((s.g, s.e, s.m, s.f), _scale(k3, dt)), p)
        ds = tuple((k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) * dt / 6 for i in range(4))
        s = State(g=max(0.0, s.g + ds[0]), e=max(0.0, s.e + ds[1]),
                  m=max(0.0, s.m + ds[2]), f=max(0.0, s.f + ds[3]))
        t += dt; states.append(State(g=s.g, e=s.e, m=s.m, f=s.f))
    return states

def compute_c3(states: List[State], p: Params) -> List[float]:
    """C3 裕度: 正=增长条件满足, 负=违规"""
    c3s = []
    for i in range(len(states)):
        s = states[i]
        dg_learn = p.effective_epsilon0(s.e, s.g) * s.e * max(0.0, 1.0 - s.g / p.g_max)
        c3 = dg_learn - p.T0 * s.f - p.delta_forget * s.g
        c3s.append(c3)
    return c3s

def _lhs_sample(bounds: dict, n: int, rng: random.Random) -> list:
    """Latin Hypercube Sampling"""
    dims = list(bounds.keys())
    per_dim = {}
    for pname in dims:
        lo, hi = bounds[pname]
        step = (hi - lo) / n
        vals = [lo + (i + rng.random()) * step for i in range(n)]
        rng.shuffle(vals)
        per_dim[pname] = vals
    return [{pname: per_dim[pname][i] for pname in dims} for i in range(n)]

_LHS_SEED = 42

def fit_params(g_obs: List[float], e_obs: List[float], n_iter: int = 80, pop_size: int = 16) -> Params:
    """差分进化拟合 ODE 参数（LHS 初始化 + 固定种子）"""
    rng = random.Random(_LHS_SEED)
    base = Params()
    # 从观测数据估算初始状态和边界
    g0_val = max(g_obs[0], 1.0); e0_val = max(e_obs[0], 1.0)
    g_max_est = max(g_obs) * 1.5; e_max_est = max(e_obs) * 1.5
    bounds = {
        "g0": (g0_val * 0.5, g0_val * 2.0), "e0": (e0_val * 0.5, e0_val * 2.0),
        "g_max": (g_max_est * 0.5, g_max_est * 3.0), "e_max": (e_max_est * 0.5, e_max_est * 3.0),
        "epsilon0": (0.01, 2.0), "alpha": (0.05, 2.0),
        "eta_ex": (0.05, 1.5), "delta_forget": (0.001, 0.3),
        "delta_f": (0.005, 0.5), "delta_entropy": (0.001, 0.2),
        "T0": (0.02, 0.8),
    }
    base = base.with_overrides(g0=g0_val, e0=e0_val, g_max=g_max_est, e_max=e_max_est)

    n_steps = len(g_obs)
    def cost(p: Params) -> float:
        try:
            p2 = p.with_overrides(t_end=n_steps * 0.5, dt=0.5)
            traj = solve_ode(p2, t_end=p2.t_end, dt=p2.dt)
            g_pred = [s.g for s in traj[:n_steps]]
            e_pred = [s.e for s in traj[:n_steps]]
            g_rmse = math.sqrt(sum((a-b)**2 for a,b in zip(g_obs, g_pred)) / n_steps)
            e_rmse = math.sqrt(sum((a-b)**2 for a,b in zip(e_obs, e_pred)) / n_steps)
            return g_rmse / (max(g_obs) or 1) + 0.3 * e_rmse / (max(e_obs) or 1)
        except:
            return 1e9

    # LHS 初始化种群
    lhs_samples = _lhs_sample(bounds, pop_size - 1, rng)
    population = [base]
    for overrides in lhs_samples:
        population.append(base.with_overrides(**overrides))

    best_params = base; best_cost = cost(base)
    no_imp = 0
    for _ in range(n_iter):
        new_pop = []
        for member in population:
            a, b = rng.sample(population, 2)
            overrides = {}
            for pname, (lo, hi) in bounds.items():
                mutant = getattr(a, pname) + rng.uniform(0.5, 1.0) * (getattr(b, pname) - getattr(a, pname))
                mutant = max(lo, min(hi, mutant))
                overrides[pname] = mutant
            cand = base.with_overrides(**overrides)
            c = cost(cand)
            if c < best_cost:
                best_cost = c; best_params = cand; no_imp = 0
            else:
                no_imp += 1
            mc = cost(member)
            new_pop.append(cand if c < mc else member)
        population = new_pop
        if no_imp >= 30: break

    return best_params


def _scale(d: tuple, f: float) -> tuple: return tuple(x * f for x in d)
def _add(a: tuple, b: tuple) -> tuple: return tuple(a[i] + b[i] for i in range(len(a)))

# ═══════════════════════════════════════════
#  CSV 列检测
# ═══════════════════════════════════════════

G_KEYWORDS = ["train_acc", "train_accuracy", "accuracy", "acc"]
E_KEYWORDS = ["val_acc", "val_accuracy", "validation_acc", "validation_accuracy"]


def detect_columns(headers: list) -> Tuple:
    headers_lower = [(i, h.strip().lower()) for i, h in enumerate(headers)]
    g_idx = g_name = e_idx = e_name = None
    for kw in G_KEYWORDS:
        for i, h in headers_lower:
            if h == kw: g_idx, g_name = i, headers[i].strip(); break
        if g_idx is not None: break
    if g_idx is None:
        for i, h in headers_lower:
            if "train_acc" in h: g_idx, g_name = i, headers[i].strip(); break
    if g_idx is None:
        for i, h in headers_lower:
            if "train_loss" in h: g_idx, g_name = i, headers[i].strip(); break
    for kw in E_KEYWORDS:
        for i, h in headers_lower:
            if h == kw: e_idx, e_name = i, headers[i].strip(); break
        if e_idx is not None: break
    if e_idx is None:
        for i, h in headers_lower:
            if "val_acc" in h: e_idx, e_name = i, headers[i].strip(); break
    if e_idx is None:
        for i, h in headers_lower:
            if "val_loss" in h: e_idx, e_name = i, headers[i].strip(); break
    return g_idx, g_name, e_idx, e_name

# ═══════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════

st.title("🔬 五环诊断 · 在线试用")
st.caption("纯 Python 引擎 | 零安装 | 上传训练日志 CSV 即刻诊断")

with st.expander("ℹ️ 这是什么？"):
    st.markdown("""
    五环诊断器基于热力学-信息论框架，用 ODE 系统拟合 ML 训练轨迹，
    输出 **C3 信息增长裕度**——一个比 loss 曲线更早发现训练无效化的指标。

    **在线版局限**：使用纯 Python ODE 引擎（与完整版 C# 诊断引擎算法等价但精度略低）。
    完整功能请 [克隆 GitHub 仓库](https://github.com/baiyaoseshi/pentacyclic) 本地运行。
    """)

uploaded = st.file_uploader("上传训练日志 CSV", type=["csv"],
    help="需要 epoch, train_acc, val_acc 列（支持常见变体）")

if uploaded:
    df = pd.read_csv(uploaded)
    headers = list(df.columns)
    g_idx, g_name, e_idx, e_name = detect_columns(headers)

    if g_idx is None or e_idx is None:
        st.error("❌ 未识别到 g 列（train_acc/loss）或 e 列（val_acc/loss）。请检查 CSV 列名。")
        st.stop()

    st.success(f"✅ g={g_name}  |  e={e_name}  |  {len(df)} 行")

    g_max = st.slider("g_max（存储容量上限）", 10.0, 1000.0, 100.0)
    e_max = st.slider("e_max（数据充裕度上限）", 10.0, 1000.0, 100.0)

    if st.button("🚀 开始诊断", type="primary"):
        with st.spinner("ODE 参数拟合中（差分进化，约 30-60 秒）..."):
            g_raw = pd.to_numeric(df.iloc[:, g_idx], errors="coerce")
            e_raw = pd.to_numeric(df.iloc[:, e_idx], errors="coerce")

            # 归一化到 [0, g_max/e_max]
            g_obs = (g_raw / g_raw.max() * g_max).tolist()
            e_obs = (e_raw / e_raw.max() * e_max).tolist()

            # 拟合
            best = fit_params(g_obs, e_obs, n_iter=80, pop_size=16)
            traj = solve_ode(best.with_overrides(t_end=len(g_obs) * 0.5, dt=0.5),
                            t_end=len(g_obs) * 0.5, dt=0.5)
            c3s = compute_c3(traj, best)

        st.subheader("📊 诊断结果")

        # 参数
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("ε₀（充能效率）", f"{best.epsilon0:.4f}")
        with col2: st.metric("α（信息转化率）", f"{best.alpha:.4f}")
        with col3: st.metric("η_ex（㶲再生效率）", f"{best.eta_ex:.4f}")

        # g 拟合对比
        st.subheader("g（储存能量）: 观测 vs 拟合")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        n = len(g_obs)
        ax1.plot(range(n), g_obs, "o", markersize=3, alpha=0.6, label="观测", color="#1f77b4")
        ax1.plot(range(n), [s.g for s in traj[:n]], "-", linewidth=2, label="拟合", color="#ff7f0e")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("g"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(range(n), e_obs, "o", markersize=3, alpha=0.6, label="观测", color="#1f77b4")
        ax2.plot(range(n), [s.e for s in traj[:n]], "-", linewidth=2, label="拟合", color="#2ca02c")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("e"); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)

        # C3 裕度
        st.subheader("C3 裕度（信息增长充要条件）")
        fig2, ax3 = plt.subplots(figsize=(12, 3))
        t_vals = list(range(len(c3s)))
        for i in range(len(t_vals)-1):
            c = "#2ca02c" if c3s[i] >= 0 else "#d62728"
            ax3.plot(t_vals[i:i+2], c3s[i:i+2], "-", color=c, linewidth=2)
        ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
        avg_c3 = sum(c3s) / len(c3s); viol = sum(1 for c in c3s if c < 0) / len(c3s) * 100
        ax3.set_title(f"C3 均值={avg_c3:.3f} | 违规 {viol:.1f}%"); ax3.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2)

        if viol > 10:
            st.warning(f"⚠️ C3 裕度违规 {viol:.1f}%——训练可能已进入无效阶段。建议调整学习率或增加数据多样性。")
        else:
            st.success("✅ C3 裕度整体健康，训练仍在有效增长区间。")

        st.caption("🔒 完整版（6 参数 · 五变量演化 · 前向预测 · 多实验对比）→ [GitHub](https://github.com/baiyaoseshi/pentacyclic)")

st.divider()
st.caption("Powered by [Pentacyclic Diagnostics](https://github.com/baiyaoseshi/pentacyclic) · Apache 2.0")
