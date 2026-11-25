"""Check v5, task 2 addendum: how far the lower boundary must go before the two boundary-sensitive rows settle."""
import pandas as pd
from multiprocessing import Pool
from check_v5_domain import P, hur, ends
from feasible import x_at_theta
def job(a):
    psi, rf, xlo = a; p = P(psi, rf); x = x_at_theta(p); n = int(round((7 - xlo) / (14 / 800))) + 1
    D0, R0 = ends(p, xlo)
    try: return dict(psi=psi, rf=rf, x_lo=xlo, n=n, D0=D0, kE=hur(p, x, "U", x_lo=xlo, n=n), kL=hur(p, x, "F", x_lo=xlo, n=n))
    except Exception as e: return dict(psi=psi, rf=rf, x_lo=xlo, n=n, err=str(e)[:50])
if __name__ == "__main__":
    jobs = [(psi, rf, xl) for psi, rf in [(2.34, .10), (11.2, .15), (2.0, .10)] for xl in (-7, -8, -9, -10, -12, -15, -20)]
    with Pool(12) as pool: d = pd.DataFrame(pool.map(job, jobs))
    s = d.to_string(index=False, float_format=lambda v: f"{v:.5f}"); print(s)
    open("output/check_v5_domain_far.txt", "w").write(__doc__ + "\n" + s + "\n")
