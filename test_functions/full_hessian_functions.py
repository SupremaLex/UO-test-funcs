from .test_function import *
from .support_funcs import *
import sympy as sp
import numpy as np


def full_hessian_1(n):
    name = "Full Hessian FH1 function"
    f = lambda x: (x[1] - 3) ** 2
    sm = lambda x, i: (x[1] - 3 - 2 * sum([x[j] for j in range(1, i + 1)])**2)**2
    x0 = np.ones((n, 1)) * 0.01
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n + 1))


def full_hessian_2(n):
    name = "Full Hessian FH2 function"
    f = lambda x: (x[1] - 5) ** 2
    sm = lambda x, i: (sum([x[j] for j in range(1, i + 1)]) - 1)**2
    x0 = np.ones((n, 1)) * 0.01
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def full_hessian_3(n):
    name = "Full Hessian FH3 function"
    f = lambda x: sum([x[i] for i in range(1, n + 1)]) ** 2
    sm = lambda x, i: (x[i] * sp.exp(x[i]) - 2*x[i] - x[i]**2)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)