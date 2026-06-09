"""
五环诊断器 · HuggingFace Space
纯 Python ODE 引擎 | 上传训练日志 CSV → 参数拟合 → C3 裕度诊断
"""
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import random
import math
from dataclasses import dataclass
from typing import List, Tuple

st.set_page_config(page_title="Pentacyclic Diagnostics", page_icon="🔬", layout="wide")

# ═══════════════════════════════════════════
#  ODE 核心
# ═══════════════════════════════════════════

@dataclass
class Params:
    epsilon0: float = 0.5; alpha: float = 0.8; sigma: float = 0.3
    eta_ex: float = 0.55; delta_forget: float = 0.02; delta_f: float = 0.05
    delta_entropy: float = 0.01; T0: float = 0.2
    g_max: float = 100.0; e_max: float = 100.0
    g0: float = 5.0; e0: float = 80.0; m0: float = 1.0; f0: float = 0.1
    t_end: float = 50.0; dt: float = 0.5

    def with_overrides(self, **kw):
        import copy
        c = copy.copy(self)
        for k, v in kw.items(): setattr(c, k, v)
        return c

@dataclass
class State:
    g: float; e: float; m: float; f: float

def ode(t: float, s: Tuple, p: Params) -> Tuple:
    g, e, m, f = s
    T0c = 1.0 / (1.0 + p.T0 * f)
    # C#-style dg
    dg_learn_raw = p.epsilon0 * e * max(0.0, 1.0 - g / p.g_max) * (0.7 + 0.3 * p.alpha)
    dg_learn = dg_learn_raw * T0c
    dg_forget = p.delta_forget * g * max(0.0, 1.0 - 0.5 * min(1.0, g / p.g_max))
    dg = dg_learn - dg_forget
    # C#-style de: F4 simplified — logistic saturation, NOT coupled to f
    de = p.eta_ex * max(0.0, 1.0 - e / p.e_max) - dg_learn
    # F2: info gain
    dm = p.alpha * dg_learn - p.delta_entropy * f * m
    # F3: entropy
    df = p.sigma * abs(dm) - p.delta_f * f
    return (dg, de, dm, df)

def solve_ode(p: Params, t_end=None, dt=None) -> List[State]:
    te = t_end or p.t_end; d = dt or p.dt
    s = State(g=p.g0, e=p.e0, m=p.m0, f=p.f0)
    states = [State(g=s.g, e=s.e, m=s.m, f=s.f)]
    t = 0.0
    for _ in range(int(te / d)):
        k1 = ode(t, (s.g, s.e, s.m, s.f), p)
        k2 = ode(t+d/2, _add((s.g,s.e,s.m,s.f), _mul(k1,d/2)), p)
        k3 = ode(t+d/2, _add((s.g,s.e,s.m,s.f), _mul(k2,d/2)), p)
        k4 = ode(t+d,   _add((s.g,s.e,s.m,s.f), _mul(k3,d)), p)
        ds = tuple((k1[i]+2*k2[i]+2*k3[i]+k4[i])*d/6 for i in range(4))
        s = State(g=max(0,s.g+ds[0]), e=max(0,s.e+ds[1]),
                   m=max(0,s.m+ds[2]), f=max(0,s.f+ds[3]))
        t += d; states.append(State(g=s.g, e=s.e, m=s.m, f=s.f))
    return states

def c3_margin(states: List[State], p: Params) -> List[float]:
    c3s = []
    for s in states:
        dg_learn = p.epsilon0 * s.e * max(0, 1 - s.g/p.g_max) / (1 + p.T0 * s.f)
        c3s.append(dg_learn - p.T0 * s.f - p.delta_forget * s.g)
    return c3s

# ── LHS + DE Fitter ──

_LHS_SEED = 42

def _lhs(bounds: dict, n: int, rng: random.Random) -> list:
    dims = list(bounds.keys()); pdim = {}
    for pn in dims:
        lo, hi = bounds[pn]; step = (hi-lo)/n
        vals = [lo+(i+rng.random())*step for i in range(n)]
        rng.shuffle(vals); pdim[pn] = vals
    return [{pn: pdim[pn][i] for pn in dims} for i in range(n)]

def fit(g_obs: List[float], e_obs: List[float], n_iter=100, pop=16) -> Params:
    """差分进化拟合（LHS 初始化 + 固定种子），同时拟合 g 和 e"""
    rng = random.Random(_LHS_SEED)
    base = Params()
    g0v = max(g_obs[0], 1); e0v = max(e_obs[0], 1)
    gme = max(g_obs)*1.5; eme = max(e_obs)*1.5
    bounds = {
        "g0": (g0v*0.5, g0v*5), "e0": (e0v*0.5, e0v*3),
        "g_max": (gme*0.5, gme*3), "e_max": (eme*0.5, eme*3),
        "epsilon0": (0.01, 2), "alpha": (0.05, 2), "eta_ex": (0.05, 1.5),
        "delta_forget": (0.001, 0.3), "delta_f": (0.005, 0.5),
        "delta_entropy": (0.001, 0.2), "T0": (0.02, 0.8),
    }
    base = base.with_overrides(g0=g0v, e0=e0v, g_max=gme, e_max=eme)
    ns = len(g_obs)
    def cost(p: Params) -> float:
        try:
            p2 = p.with_overrides(t_end=ns*0.5, dt=0.5)
            tr = solve_ode(p2, t_end=p2.t_end, dt=p2.dt)
            if len(tr) < ns: return 1e9
            gp = [s.g for s in tr[:ns]]; ep = [s.e for s in tr[:ns]]
            g_rmse = math.sqrt(sum((a-b)**2 for a,b in zip(g_obs,gp))/ns)
            e_rmse = math.sqrt(sum((a-b)**2 for a,b in zip(e_obs,ep))/ns)
            return g_rmse/(max(g_obs) or 1) + 0.3*e_rmse/(max(e_obs) or 1)
        except: return 1e9
    lhs_s = _lhs(bounds, pop-1, rng)
    population = [base] + [base.with_overrides(**o) for o in lhs_s]
    best_p = base; best_c = cost(base); noimp = 0
    for _ in range(n_iter):
        new_pop = []
        for m in population:
            a, b = rng.sample(population, 2)
            ov = {}
            for pn, (lo, hi) in bounds.items():
                mut = getattr(a,pn) + rng.uniform(0.5,1)*(getattr(b,pn)-getattr(a,pn))
                ov[pn] = max(lo, min(hi, mut))
            cand = base.with_overrides(**ov); c = cost(cand)
            if c < best_c: best_c = c; best_p = cand; noimp = 0
            else: noimp += 1
            mc = cost(m); new_pop.append(cand if c < mc else m)
        population = new_pop
        if noimp >= 30: break
    return best_p

def _mul(d, f): return tuple(x*f for x in d)
def _add(a, b): return tuple(a[i]+b[i] for i in range(len(a)))

# ═══════════════════════════════════════════
#  CSV detection
# ═══════════════════════════════════════════

G_KEYS = ["train_acc","train_accuracy","accuracy","acc"]
E_KEYS = ["val_acc","val_accuracy","validation_acc","validation_accuracy"]

def detect_cols(headers):
    hl = [(i, h.strip().lower()) for i, h in enumerate(headers)]
    gi = gn = ei = en = None
    for kw in G_KEYS:
        for i, h in hl:
            if h == kw: gi, gn = i, headers[i].strip(); break
        if gi is not None: break
    if gi is None:
        for i, h in hl:
            if "train_acc" in h: gi, gn = i, headers[i].strip(); break
    if gi is None:
        for i, h in hl:
            if "train_loss" in h or "train/loss" in h: gi, gn = i, headers[i].strip(); break
    for kw in E_KEYS:
        for i, h in hl:
            if h == kw: ei, en = i, headers[i].strip(); break
        if ei is not None: break
    if ei is None:
        for i, h in hl:
            if "val_acc" in h: ei, en = i, headers[i].strip(); break
    if ei is None:
        for i, h in hl:
            if "val_loss" in h or "val/loss" in h: ei, en = i, headers[i].strip(); break
    return gi, gn, ei, en

# ═══════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════

st.title("🔬 Pentacyclic Diagnostics · Online Demo")
st.caption("Pure Python ODE engine | Upload training log CSV | Zero install")

with st.expander("ℹ️ About"):
    st.markdown("""
    This tool fits a 5-variable thermodynamic-information ODE to your ML training logs,
    and outputs the **C3 margin** — an early-warning indicator of when training stops being effective,
    often before the loss curve shows it.

    **Limitations**: Uses a pure-Python ODE engine. For the full C# diagnostic engine with 6 parameters,
    5-variable evolution charts, forward prediction, and multi-experiment comparison,
    [clone the full version](https://github.com/baiyaoseshi/pentacyclic).
    """)

uploaded = st.file_uploader("Upload training log CSV", type=["csv"],
    help="Expected columns: epoch, train_acc, val_acc (or train_loss, val_loss)")

if uploaded:
    df = pd.read_csv(uploaded)
    headers = list(df.columns)
    gi, gn, ei, en = detect_cols(headers)

    if gi is None or ei is None:
        st.error("Could not identify g column (train_acc/loss) or e column (val_acc/loss). Check CSV headers.")
        st.stop()

    st.success(f"Detected: g={gn}  |  e={en}  |  {len(df)} rows")

    g_raw = pd.to_numeric(df.iloc[:, gi], errors="coerce").dropna()
    e_raw = pd.to_numeric(df.iloc[:, ei], errors="coerce").dropna()
    # Align lengths
    min_len = min(len(g_raw), len(e_raw))
    g_raw = g_raw.iloc[:min_len]; e_raw = e_raw.iloc[:min_len]

    g_max = st.slider("g_max (capacity ceiling)", 10.0, 1000.0, 100.0)
    e_max = st.slider("e_max (data richness ceiling)", 10.0, 1000.0, 100.0)

    if st.button("Run Diagnosis", type="primary"):
        with st.spinner("Fitting ODE parameters (differential evolution, ~30-60s)..."):
            # ── 归一化：对齐社区版逻辑 ──
            g_is_loss = gn and "loss" in gn.lower()
            e_is_loss = en and "loss" in en.lower()

            if g_is_loss:
                # loss 下降 → g 上升
                g_obs = ((1 - g_raw / g_raw.iloc[0]) * g_max).tolist()
            else:
                g_obs = (g_raw / g_raw.max() * g_max).tolist()

            if e_is_loss:
                # val_loss 初始值高 → e 可用学习空间大，下降 → e 减小
                e_obs = (e_raw / e_raw.iloc[0] * e_max).tolist()
            else:
                e_obs = (e_raw / e_raw.max() * e_max).tolist()

            best = fit(g_obs, e_obs, n_iter=100, pop=16)
            traj = solve_ode(best.with_overrides(t_end=len(g_obs)*0.5, dt=0.5),
                            t_end=len(g_obs)*0.5, dt=0.5)
            c3s = c3_margin(traj, best)

        st.subheader("Fitted Parameters")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("epsilon0 (learning efficiency)", f"{best.epsilon0:.4f}")
        with c2: st.metric("alpha (info conversion)", f"{best.alpha:.4f}")
        with c3: st.metric("eta_ex (recovery efficiency)", f"{best.eta_ex:.4f}")

        # ── g/e 对比图 ──
        st.subheader("g / e: observed vs fitted")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        n = len(g_obs)
        ax1.plot(range(n), g_obs, "o", ms=3, alpha=0.6, label="observed", color="#1f77b4")
        ax1.plot(range(n), [s.g for s in traj[:n]], "-", lw=2, label="fitted", color="#ff7f0e")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("g"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_title("g")

        ax2.plot(range(n), e_obs, "o", ms=3, alpha=0.6, label="observed", color="#1f77b4")
        ax2.plot(range(n), [s.e for s in traj[:n]], "-", lw=2, label="fitted", color="#2ca02c")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("e"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_title("e")
        plt.tight_layout(); st.pyplot(fig)

        # ── C3 裕度 ──
        st.subheader("C3 Margin (info-growth sufficiency)")
        fig2, ax3 = plt.subplots(figsize=(12, 3))
        tv = list(range(len(c3s)))
        for i in range(len(tv)-1):
            c = "#2ca02c" if c3s[i] >= 0 else "#d62728"
            ax3.plot(tv[i:i+2], c3s[i:i+2], "-", color=c, lw=2)
        ax3.axhline(0, color="gray", ls="--", alpha=0.5)
        avg = sum(c3s)/len(c3s); viol = sum(1 for c in c3s if c<0)/len(c3s)*100
        ax3.set_title(f"C3 mean={avg:.3f}  |  violation {viol:.1f}%"); ax3.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2)

        if viol > 10:
            st.warning(f"C3 violation {viol:.1f}% — training may be entering ineffective phase. Consider adjusting learning rate or increasing data diversity.")
        else:
            st.success("C3 margin is healthy — training is still in effective growth zone.")

        st.caption("[Full version](https://github.com/baiyaoseshi/pentacyclic) — 6 params · 5-variable evolution · forward prediction · multi-experiment comparison")

st.divider()
st.caption("Powered by [Pentacyclic Diagnostics](https://github.com/baiyaoseshi/pentacyclic) · Apache 2.0")
