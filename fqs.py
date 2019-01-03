# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:25:52 2019

@author: NKrvavica
"""

import numpy as np


def quartic_roots(p, complex_roots=True):
    '''
    Quartic (4th order polynomial) root solver. If polynomial has all four
    distinct and real roots, a fast closed-form analytical solver is used. If
    polynomial has multiple roots or any complex roots, a numerical eigenvalue
    solver is used (``numpy.linalg.eigvals``).
    Closed form analytical solver is about 20-30 times faster then the
    eigenvalue solver.

    Parameters
    ----------
    p: array_like
        Input data, in any form that can be converted to an array of an
        appropriate size. Array should hold coefficients of a 4th order
        polynomial of the form::

            p[0]*x^4 + p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4] = 0

        Stacked array of coefficient is allowed, which means that ``p`` may
        have size ``(5,)``, ``(5, M)`` or ``(M, 5)``, where ``M>0`` is the
        number of polynomials.

    complex_roots: bool, optional
        If set to ``True`` (default) complex roots will be computed using
        eigensolver

        If set to ``False`` complex roots will be ignored (much faster
        when only real roots are needed).

    Returns
    -------
    roots: ndarray
        Array of roots of given polynomials, size ``(M, 4)``. Real roots are
        returned sorted in ascending order.

    idx: boolean array
        Boolean mask array, ``True`` denotes indices where all four roots are
        distinct and real.

    Examples
    --------
    >>> r, idx = quartic_roots([1, 7, -806, -1050, 38322])
    >>> r
    array([[-30.76994812,  -7.60101564,   6.61999319,  24.75097057]])
    >>> idx
    array([ True])

    >>> r, idx = quartic_roots(np.array([[1, 7, -806, -1050, 38322],
                                         [0.1, -0.84, -141.5, 864, 34067]]))
    >>> r
    array([[-30.76994812,  -7.60101564,   6.61999319,  24.75097057],
       [-32.9715305 , -14.11692655,  21.59336971,  33.89508734]])
    >>> idx
    array([ True,  True])

    >>> r, idx = quartic_roots([[1, 7, -806, -1050, 38322],
                                [1, 2, 3, 4, 5],
                                [0.1, -0.84, -141.5, 864, 34067]])
    >>> r
    array([[-30.76994812+0.j        ,  -7.60101564+0.j        ,
              6.61999319+0.j        ,  24.75097057+0.j        ],
           [  0.28781548+1.41609308j,   0.28781548-1.41609308j,
             -1.28781548+0.85789676j,  -1.28781548-0.85789676j],
           [-32.9650952 +0.j        , -14.1186376 +0.j        ,
             21.60573292+0.j        ,  33.87799988+0.j        ]])
    >>> idx
    array([ True, False,  True])

    >>> r, idx = quartic_roots([[1, 7, -806, -1050, 38322],
                                [1, 2, 3, 4, 5],
                                [0.1, -0.84, -141.5, 864, 34067]],
                                complex_roots=False)
    >>> r
    array([[-30.76994812,  -7.60101564,   6.61999319,  24.75097057],
           [  0.        ,   0.        ,   0.        ,   0.        ],
           [-32.9650952 , -14.1186376 ,  21.60573292,  33.87799988]])
    >>> idx
    array([ True, False,  True])
    '''

    def real_roots(a, b, c, d, A, B, d0, d1, d0_3):
        '''A fast closed-form analytical solver for real roots of a quartic'''
        third = 1./3.
        fi = np.arccos(d1 * 0.5 / np.sqrt(d0_3))
        Z_2 = third * (2 * np.sqrt(d0) * np.cos(third * fi) - A)
        Z = np.sqrt(Z_2)
        B_Z = B / Z
        sqrt1 = np.sqrt(-A - Z_2 + B_Z)
        sqrt2 = np.sqrt(-A - Z_2 - B_Z)
        a025 = - 0.25 * a
        roots = np.array([a025 - 0.5 * (Z + sqrt1),
                          a025 - 0.5 * (Z - sqrt1),
                          a025 + 0.5 * (Z - sqrt2),
                          a025 + 0.5 * (Z + sqrt2)])
        return roots

    def eig_roots(a, b, c, d):
        '''A numerical eigenvalue solver for roots of a quartic'''
        N = len(a)
        A = np.zeros((N, 4, 4))
        A[:, 1:, :3] = np.eye(3)
        A[:, :, 3] = - np.array([d, c, b, a]).T
        roots = np.linalg.eigvals(A)
        return roots

    # convert to array (if input is a list or tuple)
    p = np.asarray(p)

    # if only one set of coefficients is given, add axis
    if p.ndim < 2:
        p = p[np.newaxis, :]

    # check if five coefficients are given
    if p.shape[1] != 5:
        if p.shape[0] == 5:
            p = p.T
        else:
            raise ValueError('Expected 4th order polynomial with 5 '
                             'coefficients, got {:d}.'.format(p.shape[1]))

    ''' compute coefficients of a corresponding polynomial of the form:
    x^4 + ax^3 + bx^2 + cx + d = 0'''
    a, b, c, d = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0], p[:, 3]/p[:, 0],
                  p[:, 4]/p[:, 0])

    # compute the discriminant and check if roots are distinct and real
    d0 = b**2 + 12*d - 3*a*c
    d0_3 = d0*d0*d0
    d1 = 27*a**2*d - 9*a*b*c + 2*b*b*b - 72*b*d + 27*c**2
    A = -0.75 * a**2 + 2 * b
    B = 0.25 * a*a*a - a * b + 2 * c
    D = 64*d - 16*b**2 + 16*a**2*b - 16*a*c - 3*(a**2)**2
    real_idx = (4*d0_3 - d1**2 > 0) & (A < 0) & (D < 0)

    roots = np.zeros((p.shape[0], 4))
    if real_idx.any():
        roots[real_idx, :] = real_roots(a[real_idx], b[real_idx], c[real_idx],
                                        d[real_idx], A[real_idx], B[real_idx],
                                        d0[real_idx], d1[real_idx],
                                        d0_3[real_idx]).T
    if (~real_idx).any() and complex_roots:
        roots = roots.astype(complex)
        roots[~real_idx, :] = eig_roots(a[~real_idx], b[~real_idx],
                                        c[~real_idx], d[~real_idx])

    return roots, real_idx
