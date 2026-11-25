"""Post-hoc additions to UQ level 0 (UQ_PLAN amendments D2-D3): kappa*/r, protocol-like states, prior on r directly."""
import numpy as np, pandas as pd
import uq0
d = pd.read_parquet("output/uq0_draws.parquet")
print("kappa*/r quantiles 5/25/50/75/95%:", np.round((d.kstar / d.r).quantile([.05, .25, .5, .75, .95]).values, 2))
print("implied r quantiles 5/50/95%:", np.round(d.r.quantile([.05, .5, .95]).values, 3), " share r>0.15:", round(float((d.r > .15).mean()), 2))
print("lemma: min over draws of kstar - r - Gamma/(1-Theta) =", f"{float((d.kstar - d.r - d.Gamma/(1-d.Theta)).min()):.1e}", "(numerical error; analytically = alpha*D/(1-Theta) > 0)")
S = uq0.sobol_binned(d); print("Sobol (ddof=0):", S.sort_values(ascending=False).round(3).to_dict(), " sum", round(S.sum(), 3))
rng = np.random.default_rng(5)
print("\nprotocol-like states (Theta, r fixed; other parameters from the plan's priors):")
for lab, Th, r in [("ETH-like", .28, .03), ("AVAX-like", .50, .07), ("SOL-like", .60, .07), ("ATOM-like", .60, .15)]:
    x = uq0.draw(20000, rng); x["Theta"] = Th; x["rho"] = r * Th
    y = uq0.run(x)
    print(f"  {lab:9s} Theta={Th} r={r:.2f}: kappa* median {y.kstar.median():.3f} (90% {y.kstar.quantile(.05):.3f}-{y.kstar.quantile(.95):.3f}),"
          f" kappa*/r {(y.kstar/y.r).median():.2f}, growth-equivalent per +10pp stake {10*y.kstar.median():.1f} pp/yr; priced {len(y)/20000:.0%}")
x = uq0.draw(20000, rng); r = rng.uniform(0.03, 0.15, 20000); x["rho"] = r * x["Theta"]
y = uq0.run(x)
print(f"\nprior on r ~ U(0.03,0.15): kappa* median {y.kstar.median():.3f} (90% {y.kstar.quantile(.05):.3f}-{y.kstar.quantile(.95):.3f}); kappa*/r median {(y.kstar/y.r).median():.2f} (90% {(y.kstar/y.r).quantile(.05):.2f}-{(y.kstar/y.r).quantile(.95):.2f})")
for a, b in ((0, .1), (0, .3)):
    k = np.random.default_rng(7).uniform(a, b, len(y)); print(f"  P(payout wins), kappa ~ U({a},{b}): {np.mean(k > y.kstar.values):.3f}")
