from .test_function import *
from .support_funcs import *
from .exceptions import *

def diagonal_1(n):
    name = "Diagonal 1 function"
    sm = lambda x, i: sp.exp(x[i]) - i * x[i]
    x0 = np.ones((n, 1)) * (1 / n)
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_2(n):
    name = "Diagonal 2 function"
    sm = lambda x, i: sp.exp(x[i]) - x[i] / i
    x0 = (1 / np.arange(1, n + 1)).reshape((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_3(n):
    name = "Diagonal 3 function"
    sm = lambda x, i: sp.exp(x[i]) - i * sp.sin(x[i])
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_4(n):
    name = "Diagonal 4 function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda x, i: 0.5 * (x[2*i-1] + 100*x[2*i])**2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0)


def diagonal_5(n):
    name = "Diagonal 5 function"
    sm = lambda x, i: sp.log(sp.exp(x[i]) + sp.exp(-x[i]))
    x0 = np.ones((n, 1)) * 1.1
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_6(n):
    name = "Diagonal 6 function"
    sm = lambda x, i: sp.exp(x[i]) + (1 - x[i])
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_7(n):
    name = "Diagonal 7 function"
    sm = lambda x, i: sp.exp(x[i]) - 2 * x[i] - x[i] ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_8(n):
    name = "Diagonal 8 function"
    sm = lambda x, i: x[i] * sp.exp(x[i]) - 2 * x[i] - x[i] ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_9(n):
    name = "Diagonal 9 function"
    sm = lambda x, i: (sp.exp(x[i]) - i * x[i]) + 10000 * x[n] ** 2
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def ext_block_diagonal(n):
    name = "Extended BD1 function (Block Diagonal)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (x[2*i-1]**2 + x[2*i]**2 - 2) ** 2
    sm_2 = lambda x, i: (sp.exp(x[2*i-1]-1) - x[2*i]) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1)) * 0.1
    return create_test_function(name, n, sm, x0)


def ext_tridiagonal_1(n):
    name = "Extended Tridiagonal 1 function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (x[2*i-1] + x[2*i] - 3) ** 2
    sm_2 = lambda x, i: (x[2*i-1] - x[2*i] + 1) ** 4
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0)


def gen_tridiagonal_1(n):
    name = "Generalized Tridiagonal 1 function"
    sm_1 = lambda x, i: (x[i] + x[i+1] - 3) ** 2
    sm_2 = lambda x, i: (x[i] - x[i+1] + 1) ** 4
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def gen_tridiagonal_2(n):
    name = "Generalized Tridiagonal 2 function"
    f = lambda x: ((5 - 3*x[1] - x[2]**2) * x[1] - 3*x[2] + 1) ** 2
    sm_1 = lambda x, i: ((5 - 3*x[i] - x[i]**2) * x[1] - x[i-1] - 3*x[i+1] + 1) ** 2
    sm_2 = lambda x: ((5 - 3*x[n] - x[n]**2) * x[n] - x[n-1] + 1) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x)
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))


def ext_tridiagonal_2(n):
    name = "Extended Tridiagonal 2 function"
    sm_1 = lambda x, i: (x[i] * x[i+1] - 1) ** 2
    sm_2 = lambda x, i: 0.1 * (x[i] + 1) * (x[i+1] + 1)
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def broyden_tridiagonal(n):
    name = "Broyden Tridiagonal function"
    f = lambda x: (3 * x[1] - 2 * x[1] ** 2) ** 2
    sm_1 = lambda x, i: (3*x[i] - 2*x[i]**2 - x[i-1] - 2*x[i+1] + 1) ** 2
    sm_2 = lambda x: (3*x[n] - 2*x[n]**2 - x[n-1] + 1) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x)
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))


def perturbed_tridiagonal_quadratic(n):
    name = "Perturbed Tridiagonal Quadratic function"
    f = lambda x: x[1] ** 2
    sm = lambda x, i: i * x[i] ** 2 + (x[i-1] + x[i] + x[i+1]) ** 2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))
