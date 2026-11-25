"""T6: lifecycle hurdles with the base growth path held fixed (feedback acts only on policy-induced stake),
the lifecycle analogue of Proposition 1's centring. G is then affine in kappa: kappa* = -G(0)/slope."""
from rt import *
from multiprocessing import Pool

def ss(psi, rf, Th=0.45, rho=0.06, c=0.01, E=0.03, mu0=0.02, a=0.3):
    G = (psi - 1) / psi * (rho - c * Th); D = rf - mu0 + E - G
    return rho / Th + (G + a * D) / (1 - Th)

def job(arg):
    psi, rf = arg
    p0 = make(psi, rf, 0.0); x = x_at_theta(p0)
    tb = lambda z: theta_of_rho(rho_of(z, p0), p0)
    out = {}
    for kind in "FU":
        g0 = gap(p0, x, kind)
        g2 = gap(with_(p0, kappa=0.2), x, kind, kb=0.0, th_base=tb)
        g4 = gap(with_(p0, kappa=0.4), x, kind, kb=0.0, th_base=tb)
        sl = (g2 - g0) / 0.2
        out[kind] = (-g0 / sl, (g4 - g0) / 0.4 / sl)  # hurdle, affinity check (should be 1)
    return f"psi={psi} rf={rf}: ss={ss(psi, rf):.3f} | basefix kL={out['F'][0]:.3f} (aff {out['F'][1]:.3f}) kE={out['U'][0]:.3f} (aff {out['U'][1]:.3f}) kL/ss={out['F'][0]/ss(psi, rf):.2f}"

if __name__ == "__main__":
    cases = [(1.1, 0.10), (1.1, 0.30), (1.5, 0.15), (1.5, 0.20), (2.0, 0.20), (11.2, 0.15), (11.2, 0.20), (11.2, 0.30), (1.1, 0.06)]
    with Pool(2) as pool:
        for line in pool.imap(job, cases): print(line, flush=True)
