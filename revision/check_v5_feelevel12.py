"""Fee-level table re-solved on [-12,7] (1087 nodes, same spacing as Table 2)."""
import numpy as np, pandas as pd
from multiprocessing import Pool
import check_v5_feelevel as F
from centred import calibrated
from feasible import x_at_theta
def job(a):
    psi, rf, Fr = a; E = 0.06 - Fr
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, E=E, F_max=4 * Fr, F_min=0.0, s=0.75); x = x_at_theta(p)
    kE = F.hur(p, x, "U", x_lo=-12, n=1087)[0]; kL = F.hur(p, x, "F", x_lo=-12, n=1087)[0]
    return dict(psi=psi, rf=rf, F=Fr, kE=kE, kL=kL, ratio=kL / kE)
if __name__ == "__main__":
    cells = [(psi, rf, Fr) for psi, rf in [(2.34, .10), (2.34, .20), (11.2, .15), (11.2, .20)] for Fr in (0.03, 0.01, 0.005)]
    with Pool(12) as pool: d = pd.DataFrame(pool.map(job, cells))
    s = d.to_string(index=False, float_format=lambda v: f"{v:.5f}"); print(s)
    open("output/check_v5_feelevel12.txt", "w").write(__doc__ + "\n" + s + "\n")
