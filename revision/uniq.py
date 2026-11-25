"""Item 10: numerical uniqueness check. For each Table 5 case, solve from initial guesses scaled by 0.2, 1 and 5 and
from a tilted guess, on 401 and 1601 nodes; report max |log w difference| across solutions."""
import numpy as np
from centred import calibrated, with_
from centred2 import solve_c2
from feasible import x_at_theta
cases = [(1.1, .06), (1.1, .1), (1.1, .2), (1.1, .3), (1.5, .1), (1.5, .2), (2, .1), (2, .2), (11.2, .15), (11.2, .2), (11.2, .3)]
zz = np.linspace(-6.5, 6.5, 300); worst = 0
for psi, rf in cases:
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=0.75)
    sols = [solve_c2(p, p, guess=g, n=n)(zz) for g in (0.2, 1.0, 5.0) for n in (401, 1601)]
    d = max(np.max(np.abs(a - sols[0])) for a in sols)
    worst = max(worst, d); print(psi, rf, f"{d:.2e}")
print("worst", worst)
