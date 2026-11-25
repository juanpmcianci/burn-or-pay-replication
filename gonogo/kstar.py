"""kappa* where payout ties burn (G2 = 0 at Theta = 0.45), against the closed form
kappa* ~= r + Gamma/(1-Theta) + 1/(M (1-Theta)), M = dlogW/dD (local Gordon approximation)."""
import numpy as np
from scipy.optimize import brentq
from cht import calibrated, solve, theta_of_rho, rho_of, gamma
from sweep import at_mid, x_at_theta

def feasible(beta, r_f, k):
    try:
        solve(calibrated(beta=beta, kappa=k, F_max=0.12, F_min=0.0, r_f=r_f)); return True
    except Exception:
        return False

for beta in [1.1, 1.5, 2.0]:
    for r_f in [0.10, 0.15]:
        lo, hi = 0.0, 1.0
        for _ in range(25):
            mid = 0.5*(lo+hi)
            lo, hi = (mid, hi) if feasible(beta, r_f, mid) else (lo, mid)
        kmax = lo
        g = lambda k: at_mid(calibrated(beta=beta, kappa=k, F_max=0.12, F_min=0.0, r_f=r_f))[0]
        p0 = calibrated(beta=beta, kappa=0.0, F_max=0.12, F_min=0.0, r_f=r_f)
        x = np.array([x_at_theta(p0)]); th = 0.45
        r = rho_of(x, p0)[0]/th; G = gamma(rho_of(x, p0), np.array([th]), p0)[0]
        M0 = at_mid(p0, field="D")[0]  # total; direct ~ total - G2
        g_hi = g(kmax*0.999)
        if g_hi > 0:
            ks = brentq(g, 0.0, kmax*0.999, xtol=1e-4)
            pk = calibrated(beta=beta, kappa=ks, F_max=0.12, F_min=0.0, r_f=r_f)
            Mk = at_mid(pk, field="D")[0] - at_mid(pk)[0]
            cf = r + G/(1-th) + 1/(Mk*(1-th))
            print(f"beta={beta} r_f={r_f}: kappa_max={kmax:.3f} kappa*={ks:.3f} closed-form={cf:.3f} (r={r:.3f}, M={Mk:.1f})")
        else:
            print(f"beta={beta} r_f={r_f}: kappa_max={kmax:.3f}  G2 at kappa_max = {g_hi:.2f} < 0: payout never wins while price exists")
