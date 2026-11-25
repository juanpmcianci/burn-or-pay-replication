"""G at kappa=0 on [-7,7] vs [-12,7] for all Table 2 rows (burn-share gap, per unit fee yield)."""
from check_v5_domain import ROWS, P
from centred2 import gap2
from feasible import x_at_theta
from multiprocessing import Pool
def job(a):
    psi, rf = a; p = P(psi, rf); x = x_at_theta(p); out = [psi, rf]
    for xlo in (-7, -12):
        n = int(round((7 - xlo) / (14 / 800))) + 1
        out += [gap2(p, x, "F", 0.0, x_lo=xlo, n=n), gap2(p, x, "U", 0.0, x_lo=xlo, n=n)]
    return out
if __name__ == "__main__":
    with Pool(12) as pool: r = pool.map(job, ROWS)
    s = "psi rf  G(-7) GE(-7)  G(-12) GE(-12)\n" + "\n".join(" ".join(f"{v:.4f}" for v in row) for row in r)
    print(s); open("output/check_v5_g0.txt", "w").write(__doc__ + "\n" + s + "\n")
