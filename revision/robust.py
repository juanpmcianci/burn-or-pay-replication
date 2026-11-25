"""Revision A1: draft 0's lifecycle robustness runs, recomputed with the feasible derivative (uniform alongside)."""
import sys, numpy as np
from multiprocessing import Pool
from feasible import *
import cht
LINEAR = cht.mu_A

def at(p):
    x = x_at_theta(p)
    d = dtheta_drho(p, x)[0]
    return G_uniform(p, x)[0] / d, G_feasible(p, x)[0] / d

def job(spec):
    name, kw, K = spec
    cht.mu_A = LINEAR
    if K is not None:
        cht.mu_A = lambda th, p: p.mu0 + K * np.tanh(p.kappa * (th - p.theta_bar) / K)
    try:
        vu, vf = at(calibrated(**kw))
        return f"{name}: V_U={vu:7.2f}  V_F={vf:7.2f}"
    except Exception as e:
        return f"{name}: -- {str(e)[:50]}"

specs = []
for rf in [0.06, 0.10]:
    for k in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        specs.append((f"low-reward r=6.7% rf={rf} k={k}", dict(beta=1.1, kappa=k, rho_t=0.03, E=0.015, F_max=0.06, F_min=0.0, r_f=rf), None))
for Fmax, s in [(0.06, 0.75), (0.12, 0.3), (0.12, 2.0), (0.20, 0.75)]:
    specs.append((f"fees Fmax={Fmax} s={s}", dict(beta=1.1, kappa=0.0, F_max=Fmax, F_min=0.0, s=s, r_f=0.06), None))
for K, psi, rf in [(0.03, 1.1, 0.06), (0.08, 1.1, 0.10)]:
    for k in [0.3, 0.6, 1.0, 2.0]:
        specs.append((f"bounded K={K} psi={psi} rf={rf} k={k}", dict(beta=psi, kappa=k, F_max=0.12, F_min=0.0, r_f=rf), K))
for k in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    specs.append((f"psi=11.2 rf=0.15 k={k}", dict(beta=11.2, kappa=k, F_max=0.12, F_min=0.0, r_f=0.15), None))

if __name__ == "__main__":
    with Pool(6) as pool:
        for line in pool.map(job, specs):
            print(line, flush=True)
