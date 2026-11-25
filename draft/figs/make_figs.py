"""Figures 1-5 for the staking burn-or-pay paper (see draft/PAPER_PLAN.md, section "Figures").

Run from anywhere:  python3 draft/figs/make_figs.py
Writes fig1_theta_mu.pdf ... fig5_hetero.pdf next to this script.
Uses only existing project code and outputs; changes no existing file.
Print style: vector PDF, <= 6.5 in wide, grayscale-safe (every series differs in
line style or marker shape as well as in color), no titles (captions live in LaTeX).
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gonogo"))

from cht import calibrated, solve  # noqa: E402
from policy import run  # noqa: E402
from panel import token_types, ratio_draws  # noqa: E402

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
# kappa is ordinal: one hue (blue), light -> dark, plus distinct line styles / markers
KAPPA_STYLE = {
    0.0: dict(color="#8db8ec", ls="-", marker="o", label=r"$\kappa=0$"),
    0.05: dict(color="#2a78d6", ls="--", marker="s", label=r"$\kappa=0.05$"),
    0.1: dict(color="#0d3c75", ls="-.", marker="^", label=r"$\kappa=0.1$"),
}
# categorical identity (channels, welfare measures): palette slots 1-3 + ink
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
# lower end of the 95% interval of the within-chain annual elasticity (revision/feedata/elasticity.csv)
_e = pd.read_csv(os.path.join(ROOT, "revision", "feedata", "elasticity.csv")).set_index("kind")
EPS_LO = float(_e.loc["within_annual", "beta"] - 1.96 * _e.loc["within_annual", "se"])

W_FULL = 6.5
W_SINGLE = 3.4


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def panel_tag(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)


# ---------------------------------------------------------------- fig 1
def fig1():
    xs = np.linspace(-3, 3, 241)
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.5))
    for k, st in KAPPA_STYLE.items():
        s = solve(calibrated(beta=1.1, kappa=k, F_max=0.12, F_min=0.0))
        th, mu = s.theta(xs) * 100, s.mu(xs) * 100
        print(f"fig1 kappa={k}: Theta {th.min():.1f}-{th.max():.1f} %, mu {mu.min():+.2f}..{mu.max():+.2f} %/yr, "
              f"corr {np.corrcoef(th, mu)[0, 1]:+.2f}")
        ax.plot(th, mu, color=st["color"], ls=st["ls"], label=st["label"])
    ax.set_xlabel(r"Staking ratio $\Theta$ (%)")
    ax.set_ylabel(r"Expected price appreciation $\mu$ (%/yr)")
    ax.legend(loc="best", handlelength=2.6)
    save(fig, "fig1_theta_mu.pdf")


# ---------------------------------------------------------------- fig 2
def fig2():
    """(a) Hurdles at the reference state by case: steady state, issuance, burn share (centred feedback,
    revision/output/table_centred.csv). (b) Burn-share over issuance hurdle against the speed at which fees fall
    (revision/output/feespeed.csv)."""
    t = pd.read_csv(os.path.join(ROOT, "revision", "output", "table_centred.csv"))
    t = t[t.status == "ok"].reset_index(drop=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ys = np.arange(len(t))[::-1]
    for y, (_, r) in zip(ys, t.iterrows()):
        a.plot([min(r.kss, r.kE, r.kL), max(r.kss, r.kE, r.kL)], [y, y], color="#cfcec8", lw=1.0, zorder=1)
    a.scatter(t.kss, ys, marker="o", s=18, facecolors="white", edgecolors=INK, linewidths=0.9, zorder=3, label="closed form")
    a.scatter(t.kE, ys, marker="s", s=16, color=CAT[0], zorder=4, label="issuance")
    a.scatter(t.kL, ys, marker="D", s=16, color=CAT[1], zorder=5, label="burn share")
    a.set_yticks(ys, [(r"B, " if r.psi > 5 else r"A, ") + rf"$\psi={r.psi:g}$, $r_f={r.rf:g}$" for _, r in t.iterrows()], fontsize=6.3)
    a.grid(axis="y", visible=False)
    a.set_xlabel(r"Hurdle $\kappa^*$")
    a.legend(loc="lower right", fontsize=6.5, handletextpad=0.2)
    panel_tag(a, "(a)")
    f = pd.read_csv(os.path.join(ROOT, "revision", "output", "feeelast.csv")).dropna(subset=["eps"])
    for (psi, rf), col, mk in [((1.1, 0.10), CAT[0], "o"), ((1.5, 0.20), CAT[2], "s"), ((11.2, 0.20), CAT[1], "D")]:
        g = f[(f.psi == psi) & (f.rf == rf) & (f.eps > -4)].sort_values("eps")
        lab = ("B" if psi > 5 else "A") + rf", $\psi={psi:g}$, $r_f={rf:g}$"
        b.plot(g.eps, g["L/E"], color=col, marker=mk, ms=3.5, label=lab)
    b.axhline(1, color=INK2, lw=0.6, ls="--")
    b.set_xlim(-4, 1.0)
    b.set_xlabel("Elasticity of dollar fees to market cap")
    b.set_ylabel(r"$\kappa^*_\omega/\kappa^*_E$")
    b.legend(loc="upper right", fontsize=6.3)
    panel_tag(b, "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_channels.pdf")


# ---------------------------------------------------------------- fig 3
PROTOCOLS = [("low, low", 0.03, 0.070, "o"), ("mid, mid", 0.07, 0.159, "s"),
             ("high, mid", 0.07, 0.191, "D"), ("high, high", 0.15, 0.352, "^")]


def fig3():
    d = pd.read_parquet(os.path.join(ROOT, "uq", "output", "uq0_draws.parquet"))
    ratio = d["kstar"] / d["r"]
    q = np.percentile(ratio, [5, 50, 95])
    print(f"fig3: n={len(d)}; kappa*/r 5/50/95% = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}; "
          f"share kappa*>r = {np.mean(d.kstar > d.r):.3f}")
    fig, (a, b, c3) = plt.subplots(1, 3, figsize=(W_FULL, 2.5), gridspec_kw=dict(width_ratios=[1, 1.1, 0.9]))
    a.hist(ratio, bins=60, color="#8db8ec", edgecolor="white", linewidth=0.3)
    a.axvline(q[1], color=INK, lw=1.2, ls="-")
    for v in (q[0], q[2]):
        a.axvline(v, color=INK2, lw=0.9, ls="--")
    ymax = a.get_ylim()[1]
    a.text(q[1], ymax * 0.97, f" median {q[1]:.2f}", va="top", ha="left", fontsize=7.5, color=INK)
    a.text(q[2], ymax * 0.80, f" 95% {q[2]:.2f}", va="top", ha="left", fontsize=7.5, color=INK2)
    a.text(q[0], ymax * 0.80, f"5% {q[0]:.2f} ", va="top", ha="right", fontsize=7.5, color=INK2)
    a.set_xlabel(r"$\kappa^*/r$")
    a.set_ylabel("Draws")
    a.set_xlim(0.9, 4.2)
    a.grid(axis="x", visible=False)
    panel_tag(a, "(a)")

    sub = d.sample(n=3000, random_state=0)
    b.scatter(sub["r"], sub["kstar"], s=3, color="#2a78d6", alpha=0.18, linewidths=0, rasterized=True)
    lim = (0, 0.40)
    b.plot([0, 0.6], [0, 0.6], color=INK2, lw=0.9, ls="--")
    b.text(0.335, 0.30, r"$\kappa^*=r$", color=INK2, fontsize=7.5, ha="left", va="top")
    for name, r, k, m in PROTOCOLS:
        b.plot(r, k, marker=m, ms=5.5, color=INK, markeredgecolor="white", markeredgewidth=0.8, ls="none")
        dx, ha = (0.008, "left")
        if name == "high, mid":
            dy = 0.012
        elif name == "mid, mid":
            dy = -0.016
        else:
            dy = 0.0
        b.text(r + dx, k + dy, name, fontsize=7.5, color=INK, ha=ha, va="center")
    b.set_xlim(*lim)
    b.set_ylim(0, 0.6)
    b.set_xlabel(r"Staking reward rate $r$")
    b.set_ylabel(r"Hurdle $\kappa^*$")
    panel_tag(b, "(b)")
    lq = pd.read_csv(os.path.join(ROOT, "revision", "output", "liquid.csv"))
    c3.fill_between(lq.phi, lq.q05, lq.q95, color="#8db8ec", alpha=0.5, linewidth=0)
    c3.plot(lq.phi, lq.q50, color=INK, lw=1.2)
    c3.axhline(1.0, color=INK2, lw=0.9, ls="--")
    c3.set_xlabel(r"Liquid share of stake $\phi$")
    c3.set_ylabel(r"$\kappa^*/r$")
    c3.set_ylim(0.9, 3.0)
    panel_tag(c3, "(c)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_uq.pdf")


# ---------------------------------------------------------------- fig 4
def fig4():
    # Same construction and seeds as gonogo/panel.py __main__: ratio_draws reseeds with
    # default_rng(12345) on every call, so dropping kappa=0.05 does not change these draws.
    # Full reps kept (300 for T<1000, 100 for T=1040); runtime is about a minute.
    thetas = np.linspace(0.15, 0.8, 8)
    Ts = [90, 208, 1040]
    res = {}
    for kappa in [0.0, 0.1]:
        types = token_types(calibrated(beta=1.1, kappa=kappa, F_max=0.12, F_min=0.0), thetas)
        for T in Ts:
            dd = ratio_draws(types, reps=300 if T < 1000 else 100, T=T)
            b, g = dd[:, 0], dd[:, 1]
            ratio = b / np.abs(g)
            q = np.percentile(ratio, [5, 50, 95])
            res[(kappa, T)] = dict(b=b, g=g, ratio=ratio)
            print(f"fig4 kappa={kappa} T={T}: beta med {np.median(b):.4f}  gamma med {np.median(g):.4f} "
                  f"(-3/(T+1) = {-3 / (T + 1):.4f})  beta/|gamma| 5/50/95% = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}; "
                  f"share outside [-10,10] = {np.mean(np.abs(ratio) > 10):.3f}")

    fig, (a, bx) = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    pos0 = np.arange(len(Ts)) * 1.0
    off = 0.2
    for kappa, sgn, hatch in [(0.0, -1, None), (0.1, 1, "//////")]:
        st = KAPPA_STYLE[kappa]
        data = [np.clip(res[(kappa, T)]["ratio"], -10, 10) for T in Ts]
        bp = a.boxplot(data, positions=pos0 + sgn * off, widths=0.32, whis=(5, 95), showfliers=False,
                       patch_artist=True, medianprops=dict(color=INK, lw=1.2),
                       whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8),
                       boxprops=dict(edgecolor=INK2, lw=0.8))
        for box in bp["boxes"]:
            box.set_facecolor(st["color"] if kappa == 0.0 else "white")
            if hatch:
                box.set_hatch(hatch)
                box.set_edgecolor(st["color"])
        a.plot([], [], marker="s", ms=7, ls="none", markerfacecolor=st["color"] if kappa == 0 else "white",
               markeredgecolor=st["color"] if kappa else INK2, label=st["label"] + (" (hatched)" if hatch else ""))
    a.axhline(0, color=INK2, lw=0.6)
    a.set_xticks(pos0, [str(T) for T in Ts])
    a.set_xlabel(r"Panel length $T$ (weeks)")
    a.set_ylabel(r"$\hat\beta/|\hat\gamma|$  (clipped to $[-10,10]$)")
    a.set_ylim(-10.5, 10.5)
    a.grid(axis="x", visible=False)
    a.legend(loc="upper right", ncol=1, handletextpad=0.4)
    panel_tag(a, "(a)")

    Tc = np.geomspace(60, 1500, 200)
    bx.plot(Tc, -3 / (Tc + 1), color=INK2, lw=1.0, ls="--", label=r"$-3/(T+1)$")
    for kappa in [0.0, 0.1]:
        st = KAPPA_STYLE[kappa]
        med = [np.median(res[(kappa, T)]["g"]) for T in Ts]
        jit = 0.95 if kappa == 0 else 1.05   # small horizontal offset so both markers show
        bx.plot(np.array(Ts) * jit, med, color=st["color"], ls="none", marker=st["marker"], ms=5.5,
                markeredgecolor=INK if kappa == 0 else "white", markeredgewidth=0.6,
                label=r"median $\hat\gamma$, " + st["label"])
    bx.set_xscale("log")
    bx.set_xticks(Ts, [str(T) for T in Ts])
    bx.minorticks_off()
    bx.set_xlabel(r"Panel length $T$ (weeks)")
    bx.set_ylabel(r"Median $\hat\gamma$")
    bx.legend(loc="lower right", handletextpad=0.4)
    panel_tag(bx, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_panel.pdf")


# ---------------------------------------------------------------- fig 5
CASE_LABEL = {
    (0.3, 1.1, 0.06, 0.03, 0.02): "Baseline",
    (0.3, 1.1, 0.10, 0.03, 0.02): r"$r_f=0.10$",
    (0.3, 2.0, 0.10, 0.03, 0.02): r"$\psi=2,\ r_f=0.10$",
    (0.3, 1.1, 0.06, 0.015, 0.02): r"$F=0.015$",
    (0.3, 1.1, 0.06, 0.045, 0.02): r"$F=0.045$",
    (0.15, 1.1, 0.06, 0.03, 0.02): r"$\alpha=0.15$",
    (0.45, 1.1, 0.06, 0.03, 0.02): r"$\alpha=0.45$",
    (0.3, 1.1, 0.06, 0.03, 0.0): r"$\mu_0=0$",
}


def gate2_tie_points():
    """Run gate2/redteam/a7.py as is and parse its 'tie points' block."""
    folder = os.path.join(ROOT, "gate2", "redteam")
    outp = subprocess.run([sys.executable, "a7.py"], cwd=folder, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in outp.splitlines():
        if line.startswith("{'alpha'"):
            i = line.index("} {") + 1
            case, tie = ast.literal_eval(line[:i]), ast.literal_eval(line[i + 1:])
            rows.append((case, tie))
    return rows


def fig5():
    rows = gate2_tie_points()
    measures = [("G_M", "Market cap", INK, "o", True),
                ("G_tot", "Total welfare", CAT[0], "s", True),
                ("G_s", "Stakers", CAT[1], "^", False),
                ("G_h", "Holders", CAT[2], "D", False)]
    fig, ax = plt.subplots(figsize=(W_SINGLE + 0.6, 2.9))
    n = len(rows)
    ys = np.arange(n)[::-1]
    for y, (case, tie) in zip(ys, rows):
        key = (case["alpha"], case["psi"], case["rf"], case["F"], case["mu0"])
        vals = [tie[m] for m, *_ in measures]
        print(f"fig5 {CASE_LABEL[key]:>24s}: " + "  ".join(f"{m}={tie[m]:.3f}" for m, *_ in measures))
        ax.plot([min(vals), max(vals)], [y, y], color="#cfcec8", lw=1.0, zorder=1)
    for j, (m, lab, c, mk, filled) in enumerate(measures):
        xv = [tie[m] for _, tie in rows]
        ax.scatter(xv, ys, marker=mk, s=26, zorder=3 + j, label=lab,
                   facecolors=c if filled else "white", edgecolors=c, linewidths=1.0)
    labels = [CASE_LABEL[(c["alpha"], c["psi"], c["rf"], c["F"], c["mu0"])] for c, _ in rows]
    ax.set_yticks(ys, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"Tie point $\kappa^*$ (staker wealth share $\lambda=0.75$)")
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.4, 1.0), handletextpad=0.2, columnspacing=0.9)
    save(fig, "fig5_hetero.pdf")


# ---------------------------------------------------------------- Proposition 2 weights
def fig_weights():
    """Weights pi_E, pi_L of Proposition 2 and the local hurdle kstar(z), revision/output/prop2_weights.csv."""
    w = pd.read_csv(os.path.join(ROOT, "revision", "output", "prop2_weights.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.5), sharey=False)
    for ax, (key, lab) in zip(axes, [("A|1.5|0.2|{}", r"A, $\psi=1.5$, $r_f=0.2$"), ("B|11.2|0.2|{}", r"B, $\psi=11.2$, $r_f=0.2$")]):
        g = w[w.case == key]; g = g[(g.dz > -4) & (g.dz < 5)]
        ax.plot(g.dz, g.piE, color=CAT[0], ls="-", label=r"weight, issuance $\pi_E$")
        ax.plot(g.dz, g.piL, color=CAT[1], ls="--", label=r"weight, burn share $\pi_\omega$")
        ax.set_xlabel(r"$\log A$ relative to the reference state")
        ax.set_ylabel("Weight density")
        t = ax.twinx(); t.grid(False)
        t.plot(g.dz, g.kstar, color=INK, ls=":", lw=1.2, label=r"local hurdle $\hat\kappa(z)$")
        t.set_ylabel(r"Local hurdle $\hat\kappa(z)$")
        t.set_ylim(0, max(1.1, float(g.kstar.max()) * 1.05))
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = t.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.3)
        ax.text(0.03, 0.04, lab, transform=ax.transAxes, fontsize=7, va="bottom")
    panel_tag(axes[0], "(a)"); panel_tag(axes[1], "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_weights.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "w": fig_weights}[w]()
