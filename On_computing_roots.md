# On computing roots of quartic and cubic equations in Python



_by Nino Krvavica ([nino.krvavica@uniri.hr](nino.krvavica@uniri.hr))_



**Abstract**

This document examines various ways to compute roots of cubic (3rd order polynomial) and quartic (4th order polynomial) equations in Python. First, two numerical algorithms, available from Numpy package (`roots` and `linalg.eigvals`), were analyzed. Then, an optimized closed-form analytical solutions to cubic and quartic equations were implemented and examined. Finally, the analytical solutions were vectorized by using `numpy` arrays in order to avoid slow python iterations when multiple polynomials are solved. All these functions were evaluated by comparing their computational speeds. Analytical cubic and quartic solvers were one order of magnitude faster than both numerical Numpy functions for a single polynomial. When a large set of polynomials were given as input, the vectorized analytical solver outperformed the numerical Numpy functions by one and two orders of magnitude, respectively. 

**Keywords:** _cubic_, _quartic_, _python_, _numpy_, _closed-form_, _polynomial roots_, _eigenvalues_, _fqs_



## Introduction

In scientific computing we are sometimes faced with solving roots of a [cubic](https://en.wikipedia.org/wiki/Cubic_function) (3rd order polynomial) or [quartic](https://en.wikipedia.org/wiki/Quartic_function) equation (4th order polynomial) to get crucial information about the characteristics of some physical process or to develop an appropriate numerical scheme. These issues are regularly encountered when analyzing coupled dynamic systems described by three or four differential equations. One such example is a two-layer [Shallow Water Flow](https://en.wikipedia.org/wiki/Shallow_water_equations) (SWE), which is defined by four [Partial Differential Equations](https://en.wikipedia.org/wiki/Partial_differential_equation) (PDE). In two-layer SWE, the [eigenvalues](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of the 4x4 flux matrix describe the speed of internal and external wave propagation. And, the eigenvalues correspond to roots of a characteristic 4th order polynomial. Similarly, SWE coupled with sediment transport are defined by three PDEs. In this case,  the eigenvalues of a 3x3 matrix correspond to roots of a characteristic 3rd order polynomial. There are many more examples where such computation is required.

Roots of cubic and quartic equations can be computed using [numerical methods](https://en.wikipedia.org/wiki/Numerical_method) or analytical expressions (so called [closed-form solutions](https://en.wikipedia.org/wiki/Closed-form_expression)). Numerical methods are based on specific algorithms and provide only approximations to roots. [Root-finding algorithms](https://en.wikipedia.org/wiki/Root-finding_algorithm) (such as Newton's, secant, Brent's method, etc.) are appropriate for any continuous function, they use iterations but do not guarantee that all roots will be found. However, a different class of numerical methods is available (and recommended) for polynomials, based on finding [eigenvalues](https://en.wikipedia.org/wiki/Eigenvalue_algorithm) of the companion matrix of a polynomial. 

In Python, there are several ways to numerically compute roots of any polynomial; however, only two functions are generally recommended and used. First is a [Numpy](http://www.numpy.org/) function called `roots` which directly computes all roots of a general polynomial, and the second is also a [Numpy](http://www.numpy.org/) function from `linalg` module called `eigvals`, which computes eigenvalues of a companion matrix constructed from a given (characteristic) polynomial.

On the other hand, analytical [closed-form solutions exist for all polynomials of degree lower than five](https://en.wikipedia.org/wiki/Abel%E2%80%93Ruffini_theorem), that is, for quadratic, cubic, and quartic equations. Although, the expressions for cubic and quadratic roots are longer and more complicated than for a quadratic equations, they can still be easily implemented in some computational algorithm. The closed-form solution for roots of cubic equations is based on Cardano's expressions given [here](https://en.wikipedia.org/wiki/Cubic_function) and [here](http://www.1728.org/cubic2.htm). Similarly, solution to the roots for quartic equations is based on Ferrari's expressions given [here](https://en.wikipedia.org/wiki/Quartic) and [here.](http://www.1728.org/quartic2.htm) A fast and optimized algorithm - [FQS](https://github.com/NKrvavica/fqs) - that uses analytical solutions to cubic and quartic equation was implemented in Python and made publicly available [here](https://github.com/NKrvavica/fqs).

All computational algorithms were implemented in Python 3.7 with Numpy 1.15, and tests were done on Windows 64-bit machine, i5-2500 CPU @ 3.30 GHz.



## Numerical algorithms

### Function `numpy.roots`

Function `numpy.roots` can compute roots of a general polynomial defined by a list of its coefficients `p`.

For cubic equations, `p` is defined as:

	p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3],

and for quartic equations, `p` is defined as:

	p[0]*x**4 + p[1]*x**3 + p[2]*x**2 + p[3]*x + p[4].

The function to compute roots of a single polynomial is implemented for cubic roots as follows:

	import numpy
	
	p_cubic = numpy.random.rand(4)
	cubic_roots = numpy.roots(p_cubic)

 and for quartic roots:

	p_quartic = numpy.random.rand(5)
	quartic_roots = numpy.roots(p_quartic)

The respective results are:

	>>> p_cubic
	array([0.21129527, 0.23589228, 0.73094489, 0.84747689])
	>>> cubic_roots
	array([ 0.01557778+1.86945535j,  0.01557778-1.86945535j,
			-1.1475662 +0.j        ]))

and

	>>> p_quartic
	array([0.30022249, 0.31473263, 0.00791689, 0.06335546, 0.73838408])
	>>> quartic_roots
	array([-1.19538943+0.7660177j , -1.19538943-0.7660177j ,
			0.67122379+0.87725993j,  0.67122379-0.87725993j])

Let's look at the computation times using `timeit` function:

	import timeit
	
	%timeit numpy.roots(p_cubic)
	76.5 µs ± 148 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
	
	%timeit numpy.roots(p_quartic)
	80.1 µs ± 2.98 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

This may seem quite fast; however, what if we need to solve, let's say, 10 000 polynomials:

    p_cubics = numpy.random.rand(10_000, 4)
    p_quartics = numpy.random.rand(10_000, 5)

Notice that `numpy.roots` takes only rank-1 arrays, which means that we have to use `for` loops or [list comprehensions](https://en.wikipedia.org/wiki/List_comprehension). The latter, are usually faster in python (and more _pythonish_), therefore we write:

    cubic_roots = [numpy.roots(pi) for pi in p_cubics]
    quartic_roots = [numpy.roots(pi) for pi in p_quartics]

Their corresponding computation times:

    %timeit [numpy.roots(pi) for pi in p_cubics]
    786 ms ± 15.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    
    %timeit [numpy.roots(pi) for pi in p_quartics]
    795 ms ± 2.51 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

It seems that it takes 10 000 times longer to compute 10 000 polynomials than 1 polynomial. 

Can we speed things up, knowing that loops in Python are slower in comparison to other high level languages, such as C/C++ or FORTRAN?



### Function `numpy.linalg.eigvals`

Documentation for `numpy.roots` states that its algorithms rely on computing the eigenvalues of the _companion_ matrix. It uses the fact that the eigenvalues of a matrix `A` are the roots of its characteristic polynomial `p`. 

Function `numpy.linalg.eigvals` computes eigenvalues of a general square matrix `A` using _geev_ [LAPACK](http://www.netlib.org/lapack/) routines. The main advantage of the function `linalg.eigvals` over `roots` is that it uses [vectorization](https://en.wikipedia.org/wiki/Array_programming). Meaning, it runs certain operations over entire array, rather than over individual elements. Therefore, it can take as input stacked array of companion matrices, and does not require `for` loops or list comprehensions.

For cubic equations, first we reduce the polynomial to the form: 

	x**3 + a*x**2 +b*x + c = 0,

and then construct the companion matrix:

	A = [[0, 0, -c],
		 [1, 0, -b],
		 [0, 1, -a]]

For quartic equations, we reduce the polynomial to the form: 

	x**4 + a*x**3 + b*x**2 +c*x + d = 0,

and then construct the companion matrix:

	A = [[0, 0, 0, -d],
		 [1, 0, 0, -c],
		 [0, 1, 0, -b],
		 [0, 0, 1, -a]]

The function to compute roots from eigenvalues of a single companion matrix is implemented for cubic equation as follows:

	def eig_cubic_roots(p):
		
		# Coefficients of quartic equation
		a, b, c = p[:, 1]/p[:, 0], p[:, 2]/p[:, 0], p[:, 3]/p[:, 0]
		
		# Construct the companion matrix
		A = numpy.zeros((len(a), 3, 3))
		A[:, 1:, :2] = numpy.eye(2)
		A[:, :, 2] = -numpy.array([c, b, a]).T
		
		# Compute roots using eigenvalues
		return numpy.linalg.eigvals(A)


Similarly, for quartic equation:

	def eig_quartic_roots(p):
		
		# Coefficients of quartic equation
		a, b, c, d = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0],
					  p[:, 3]/p[:, 0], p[:, 4]/p[:, 0])
		
		# Construct the companion matrix
		A = numpy.zeros((len(a), 4, 4))
		A[:, 1:, :3] = numpy.eye(3)
		A[:, :, 3] = -numpy.array([d, c, b, a]).T
		
		# Compute roots using eigenvalues
		return numpy.linalg.eigvals(A)


To compute roots of a single cubic equation, `eigvals` is implemented as follows:

    cubic_roots = eig_cubic_roots(p_cubic[None, :])

 and for quartic roots:

	quartic_roots = eig_quartic_roots(p_quartic[None, :])

The results are:

	>>> cubic_roots
	array([[-1.1475662 +0.j        ,  0.01557778+1.86945535j,
			0.01557778-1.86945535j]])

and
```	>>> quartic_roots
>>> quartic_roots
array([[ 0.67122379+0.87725993j,  0.67122379-0.87725993j,
		-1.19538943+0.7660177j , -1.19538943-0.7660177j ]])
```

Let's look at the computation times:

	%timeit eig_cubic_roots(p_cubic[None, :])
	67 µs ± 316 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
	
	%timeit eig_quartic_roots(p_quartic[None, :])
	69.3 µs ± 135 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

The computation times are slightly faster than `numpy.roots`.

Let see the difference for 10 000 polynomials:

    %timeit eig_cubic_roots(p_cubics)
    31.2 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
    %timeit eig_quartic_roots(p_quartics)
    48.3 ms ± 48.6 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

The results indicate that `numpy.linalg.eigvals` is **one order of magnitude** faster than `numpy.roots`, and that is a huge gain. 



## What about analytical solutions to cubic and quartic equations?

### Implementation for a single polynomial

Now, let's look at analytical implementations available by [FQS](https://github.com/NKrvavica/fqs) function. First, implementation of analytical solutions for single quadratic, cubic and quartic equation is presented. As stated in the introduction, these algorithms are based on closed-form solutions for cubic (given [here](https://en.wikipedia.org/wiki/Cubic_function) and [here](http://www.1728.org/cubic2.htm)) and quartic equations (given [here](https://en.wikipedia.org/wiki/Quartic) and [here](http://www.1728.org/quartic2.htm)). These equations were modified to avoid repeating computations.

Python function for roots of a quadratic equation is implemented as:

    import math, cmath
    
    def single_quadratic(a0, b0, c0):
        ''' Reduce the quadratic equation to to form:
            x**2 + a*x + b = 0 '''
        a, b = b0 / a0, c0 / a0
    
        # Some repeating variables
        a0 = -0.5*a
        delta = a0*a0 - b
        sqrt_delta = cmath.sqrt(delta)
    
        # Roots
        r1 = a0 - sqrt_delta
        r2 = a0 + sqrt_delta
    
        return r1, r2

The function for roots of a cubic equation is implemented as:


    def single_cubic(a0, b0, c0, d0):
        ''' Reduce the cubic equation to to form:
            x**3 + a*x**2 + b*x + c = 0 '''
        a, b, c = b0 / a0, c0 / a0, d0 / a0
    
        # Some repeating constants and variables
        third = 1./3.
        a13 = a*third
        a2 = a13*a13
        sqr3 = math.sqrt(3)
    
        # Additional intermediate variables
        f = third*b - a2
        g = a13 * (2*a2 - b) + c
        h = 0.25*g*g + f*f*f
    
        def cubic_root(x):
            ''' Compute cubic root of a number while maintaining its sign'''
            if x.real >= 0:
                return x**third
            else:
                return -(-x)**third
    
        if f == g == h == 0:
            r1 = -cubic_root(c)
            return r1, r1, r1
    
        elif h <= 0:
            j = math.sqrt(-f)
            k = math.acos(-0.5*g / (j*j*j))
            m = math.cos(third*k)
            n = sqr3 * math.sin(third*k)
            r1 = 2*j*m - a13
            r2 = -j * (m + n) - a13
            r3 = -j * (m - n) - a13
            return r1, r2, r3
    
        else:
            sqrt_h = cmath.sqrt(h)
            S = cubic_root(-0.5*g + sqrt_h)
            U = cubic_root(-0.5*g - sqrt_h)
            S_plus_U = S + U
            S_minus_U = S - U
            r1 = S_plus_U - a13
            r2 = -0.5*S_plus_U - a13 + S_minus_U*sqr3*0.5j
            r3 = -0.5*S_plus_U - a13 - S_minus_U*sqr3*0.5j
            return r1, r2, r3

And finally, the function for roots of a quartic equation is implemented as:


    def single_quartic(a0, b0, c0, d0, e0):
     
        ''' Reduce the quartic equation to to form:
            x**4 + a*x**3 + b*x**2 + c*x + d = 0'''
        a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0
    
        # Some repeating variables
        a0 = 0.25*a
        a02 = a0*a0
    
        # Coefficients of subsidiary cubic equation
        p = 3*a02 - 0.5*b
        q = a*a02 - b*a0 + 0.5*c
        r = 3*a02*a02 - b*a02 + c*a0 - d
    
        # One root of the cubic equation
        z0, _, _ = single_cubic(1, p, r, p*r - 0.5*q*q)
    
        # Additional variables
        s = cmath.sqrt(2*p + 2*z0.real + 0j)
        if s == 0:
            t = z0*z0 + r
        else:
            t = -q / s
    
        # Compute roots by quadratic equations
        r0, r1 = single_quadratic(1, s, z0 + t)
        r2, r3 = single_quadratic(1, -s, z0 - t)
    
        return r0 - a0, r1 - a0, r2 - a0, r3 - a0

Notice that the quartic solution calls both cubic and quadratic functions.

For a single cubic equation, the results are:

```
>>> single_cubic(*p_cubic)
((-1.147566194142574+0j),
 (0.01557779507848811+1.8694553386446031j),
 (0.01557779507848811-1.8694553386446031j))
```

and for a quartic:

```
>>> single_quartic(*p_quartic)
((-1.195389428644198+0.766017693855147j),
 (-1.195389428644198-0.766017693855147j),
 (0.6712237840251022+0.8772599258280781j),
 (0.6712237840251022-0.8772599258280781j))
```

Let's look at the computation times:

```
%timeit single_cubic(*p_cubic)
28.6 µs ± 604 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit single_quartic(*p_quartic)
50.3 µs ± 636 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

We can notice that `single_cubic` is **twice as fast** than both numerical solvers implemented in Numpy. Whereas, `single_quartic` is about 30-40% faster than the numerical solvers.

What about multiple polynomials?

```
%timeit [single_cubic(*pi) for pi in p_cubics]
236 ms ± 7.89 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit [single_quartic(*pi) for pi in p_quartics]
421 ms ± 1.84 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Analytical closed-form solver is about **twice as fast**  than `numpy.roots`. However, it is much slower (almost one order of magnitude) than the numerical solver from `numpy.linalg.eigvals`. This difference is mainly the consequence of using list comprehension.

### Just-in-time compiler from Numba

Let's see if we can speed up the computation by using _just-in-time_ compiler from [Numba](http://numba.pydata.org/). We only need to import it and put a decorator before each function:

```
from numba import jit

@jit(nopython=True)
def single_quadratic(a0, b0, c0):
	...

@jit(nopython=True)
def single_cubic(a0, b0, c0, d0):
	...

@jit(nopython=True)
def single_quartic(a0, b0, c0, d0, e0):
	...
	
```

These are new computation times after Numba was implemented:

```
%timeit single_cubic(*p_cubic)
6.34 µs ± 7.75 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit single_quartic(*p_quartic)
5.8 µs ± 6.87 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

We can notice that the algorithms are **several times faster** than both numerical solvers implemented in Numpy. 

What about multiple polynomials?

```
%timeit [single_cubic(*pi) for pi in p_cubics]
27.6 ms ± 80.5 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit [single_quartic(*pi) for pi in p_quartics]
30.8 ms ± 90.6 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Analytical closed-form solver (with Numba) is **one order of magnitude faster** than `numpy.roots` and shows similar performance, but slightly faster, than numerical solver from `numpy.linalg.eigvals`. 

However, notice that list comprehensions were used here for multiple inputs. What if we could vectorize this code using Numpy arrays and speed up computation times even more?



### Vectorized analytical closed-form solvers

To vectorize functions `single_quadratic`, `single_cubic`, and `single_quartic` using Numpy arrays we have to get rid of all `if` clauses and replace them with Numpy masks. Also, we have to replace all mathematical functions from `math` and `cmath` with corresponding Numpy functions. This is implemented as follows.

For quadratic equation:


    def multi_quadratic(a0, b0, c0):
    	# Quadratic coefficient
        a, b = b0 / a0, c0 / a0
    
        # Some repeating variables
        a0 = -0.5*a
        delta = a0*a0 - b
        sqrt_delta = numpy.sqrt(delta + 0j)
    
        # Roots
        r1 = a0 - sqrt_delta
        r2 = a0 + sqrt_delta
    
        return r1, r2

For cubic equation:


    def multi_cubic(a0, b0, c0, d0, all_roots=True):
    	# Cubic coefficients
        a, b, c = b0 / a0, c0 / a0, d0 / a0
    
        # Some repeating constants and variables
        third = 1./3.
        a13 = a*third
        a2 = a13*a13
        sqr3 = math.sqrt(3)
    
        # Additional intermediate variables
        f = third*b - a2
        g = a13 * (2*a2 - b) + c
        h = 0.25*g*g + f*f*f
    
        # Masks for different combinations of roots
        m1 = (f == 0) & (g == 0) & (h == 0)     # roots are real and equal
        m2 = (~m1) & (h <= 0)                   # roots are real and distinct
        m3 = (~m1) & (~m2)                      # one real root and two complex
    
        def cubic_root(x):
            ''' Compute cubic root of a number while maintaining its sign
            '''
            root = numpy.zeros_like(x)
            positive = (x >= 0)
            negative = ~positive
            root[positive] = x[positive]**third
            root[negative] = -(-x[negative])**third
            return root
    
        def roots_all_real_equal(c):
            ''' Compute cubic roots if all roots are real and equal
            '''
            r1 = -cubic_root(c)
            if all_roots:
                return r1, r1, r1
            else:
                return r1
    
        def roots_all_real_distinct(a13, f, g, h):
            ''' Compute cubic roots if all roots are real and distinct
            '''
            j = numpy.sqrt(-f)
            k = numpy.arccos(-0.5*g / (j*j*j))
            m = numpy.cos(third*k)
            r1 = 2*j*m - a13
            if all_roots:
                n = sqr3 * numpy.sin(third*k)
                r2 = -j * (m + n) - a13
                r3 = -j * (m - n) - a13
                return r1, r2, r3
            else:
                return r1
    
        def roots_one_real(a13, g, h):
            ''' Compute cubic roots if one root is real and other two are complex
            '''
            sqrt_h = numpy.sqrt(h)
            S = cubic_root(-0.5*g + sqrt_h)
            U = cubic_root(-0.5*g - sqrt_h)
            S_plus_U = S + U
            r1 = S_plus_U - a13
            if all_roots:
                S_minus_U = S - U
                r2 = -0.5*S_plus_U - a13 + S_minus_U*sqr3*0.5j
                r3 = -0.5*S_plus_U - a13 - S_minus_U*sqr3*0.5j
                return r1, r2, r3
            else:
                return r1
    
        # Compute roots
        if all_roots:
            roots = numpy.zeros((3, len(a))).astype(complex)
            roots[:, m1] = roots_all_real_equal(c[m1])
            roots[:, m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
            roots[:, m3] = roots_one_real(a13[m3], g[m3], h[m3])
        else:
            roots = numpy.zeros(len(a))
            roots[m1] = roots_all_real_equal(c[m1])
            roots[m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
            roots[m3] = roots_one_real(a13[m3], g[m3], h[m3])
    
        return roots

And for quadratic equation:


    def multi_quartic(a0, b0, c0, d0, e0):
    	# Quartic coefficients
        a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0
    
        # Some repeating variables
        a0 = 0.25*a
        a02 = a0*a0
    
        # Coefficients of subsidiary cubic equation
        p = 3*a02 - 0.5*b
        q = a*a02 - b*a0 + 0.5*c
        r = 3*a02*a02 - b*a02 + c*a0 - d
    
        # One root of the cubic equation
        z0 = multi_cubic(1, p, r, p*r - 0.5*q*q, all_roots=False)
    
        # Additional variables
        s = numpy.sqrt(2*p + 2*z0.real + 0j)
        t = numpy.zeros_like(s)
        mask = (s == 0)
        t[mask] = z0[mask]*z0[mask] + r[mask]
        t[~mask] = -q[~mask] / s[~mask]
    
        # Compute roots by quadratic equations
        r0, r1 = solve_multi_quadratic(1, s, z0 + t) - a0
        r2, r3 = solve_multi_quadratic(1, -s, z0 - t) - a0
    
        return r0, r1, r2, r3

Let's examine the computation time of vectorized analytical closed-form solvers for a single polynomial:

```
%timeit multi_cubic(*p_cubic.T)
174 µs ± 2.02 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit multi_quartic(*p_quartic.T)
233 µs ± 4.91 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

Clearly, vectorized version for a single polynomial is overkill, and results in slowest computation times.

But what about multiple polynomials:


```
%timeit multi_cubic(*p_cubics.T)
3.14 ms ± 13.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit multi_quartic(*p_quartics.T)
5.46 ms ± 38.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

The vectorized implementation of analytical solvers is an order of magnitude faster than original analytical implementation, even with _just-in-time_ compiler from Numba. Furthermore, the vectorized implementation is two order of magnitudes faster than the numerical solver `roots` and also one-order of magnitude faster than `linalg.eigvals`.



## Summary (TL;DR)

Findings on computation speed of different ways to solve cubic and quartic equations in Python can be summarized as follows:

* Two numerical algorithms for finding polynomial roots are available out-of-box from Numpy package (`numpy.roots` and `numpy.linalg.eigvals`)
* Analytical algorithms (closed-form solutions) for solving polynomial roots were implemented in Python (`single_cubic/single_quartic` for a single polynomial, and vectorized `multi_cubic/multi_quartic`  for multiple polynomials). These functions are available through [FQS](https://github.com/NKrvavica/fqs)
* Both numerical algorithms have similar CPU times for a single polynomial, but for multiple polynomials `linalg.eigvals` becomes much faster (up to one order of magnitude)
* Analytical algorithm `single_cubic/single_quartic` is the fastest when a single polynomial, or a set smaller then 100 polynomials should be solved
* For `single_cubic/single_quartic` _just-in-time_ compiler from Numba gives a significant increase in the computational speed
* Analytical algorithm `multi_cubic/multi_quartic` is the fastest when a set larger than 100 polynomials is given as input
* A Python function containing `single_cubic` , `single_quartic`, `multi_cubic`, and `multi_quartic`, as well as a function than determines what solver should be used in a specific case, is available through [FQS](https://github.com/NKrvavica/fqs).

The CPU times are summarized in the following two tables for different number of polynomials (Nr.) and separately for cubic and quartic equations:

|   Nr. | `roots` | `linalg.eigvals` | `single_cubic` | `single_cubic(@jit)` | `multi_cubic` |
| ----: | ------- | ---------------- | -------------- | -------------------- | ------------- |
|     1 | 76.5 µs | 67 µs            | 28.6 µs        | **6.34 µs**          | 174 µs        |
|   100 | 8.19 ms | 0.54 ms          | 2.11 ms        | **0.27 ms**          | **0.24 ms**   |
| 10000 | 786 ms  | 31.2 ms          | 236 ms         | 27.6 ms              | **3.14 ms**   |

|   Nr. | `roots` | `linalg.eigvals` | `single_quartic` | `single_quartic (@jit)` | `multi_quartic` |
| ----: | ------- | ---------------- | ---------------- | ----------------------- | --------------- |
|     1 | 80.1 µs | 69.3 µs          | 50.3 µs          | **5.8 µs**              | 233 µs          |
|   100 | 8.22 ms | 0.59 ms          | 3.94 ms          | **0.33 ms**             | **0.34 ms**     |
| 10000 | 795 ms  | 48.3 ms          | 421 ms           | 30.8 ms                 | **5.46 ms**     |

