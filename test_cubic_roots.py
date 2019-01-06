# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:14:52 2019

@author: NKrvavica
"""

import timeit
import numpy as np
import fqs


def numpy_roots(p):
    '''Find roots of a polynomial by `numpy.roots`.'''
    roots = []
    for pi in p:
        roots.append(np.roots(pi))
    return roots


def eig_roots(p):
    '''Finds cubic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 3x3 companion matrix'''
    a, b, c = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0], p[:, 3]/p[:, 0],)
    N = len(a)
    A = np.zeros((N, 3, 3))
    A[:, 1:, :2] = np.eye(2)
    A[:, :, 2] = - np.array([c, b, a]).T
    roots = np.linalg.eigvals(A)
    return roots


# --------------------------------------------------------------------------- #
# Test speed of fqs cubic solver compared to np.roots and np.linalg.eigvals
# --------------------------------------------------------------------------- #

# Number of samples (sets of randomly generated quartic coefficients)
N = 10000

# Generate polynomial coefficients
range_coeff = 100
p = np.random.rand(N, 4)*(range_coeff) - range_coeff/2

# number of runs
runs = 5

best_time = 100
for i in range(runs):
    start = timeit.default_timer()
    roots1 = numpy_roots(p)
    stop = timeit.default_timer()
    time = stop - start
    best_time = min(best_time, time)
print('np.roots: {:.2f} ms (best of {} runs)'.format(best_time*1000, runs))

best_time = 100
for i in range(runs):
    start = timeit.default_timer()
    roots2 = eig_roots(p)
    stop = timeit.default_timer()
    time = stop - start
    best_time = min(best_time, time)
print('np.linalg.eigvals: {:.2f} ms (best of {} runs)'.format(best_time*1000,
      runs))
print('max err: ', (abs(np.sort(roots2, axis=1)
                    - (np.sort(roots1, axis=1)))).max())

best_time = 100
for i in range(runs):
    start = timeit.default_timer()
    roots3 = fqs.cubic_roots(p)
    stop = timeit.default_timer()
    time = stop - start
    best_time = min(best_time, time)
print('fqs.cubic_roots: {:.2f} ms (best of {} runs)'.format(best_time*1000,
      runs))
print('max err: ', (abs(np.sort(roots3, axis=1)
                    - (np.sort(roots1, axis=1)))).max())
# --------------------------------------------------------------------------- #
