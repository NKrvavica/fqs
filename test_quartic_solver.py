# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:25:52 2019

@author: NKrvavica
"""

import timeit
import numpy as np
from fqs import quartic_roots


def generate_real_poly_Coeff(N):
    x1, x2, x3, x4 = np.random.rand(N, 4).T*100
    a = - (x1 + x2 + x3 + x4)
    b = x1 * x2 + (x1 + x2)*(x3 + x4) + x3 * x4
    c = - x1 * x2 * (x3 + x4) - x3 * x4 * (x1 + x2)
    d = x1 * x2 * x3 * x4
    p = np.array([np.ones(N), a, b, c, d]).T
    return p


# Number of samples (sets of quartic coefficients)
N = 100000

# Generate a large set of quartic coefficients with all real roots
p_real = generate_real_poly_Coeff(N)
start = timeit.default_timer()
r1, idx1 = quartic_roots(p_real)
stop = timeit.default_timer()
time1 = stop - start
print(time1)

# Generate a large set of random quartic coefficients (mostly complex roots)
p_rand = np.random.rand(N, 5)
start = timeit.default_timer()
r2, idx2 = quartic_roots(p_rand)
stop = timeit.default_timer()
time2 = stop - start
print(time2)
