from .test_function import *
from .support_funcs import *
from .exceptions import *

def diagonal_1(n):
    name = "Diagonal 1 function"
    sm = lambda i: sp.exp(xi(i)) - i * xi(i)
    x0 = np.ones((n, 1)) * (1 / n)
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_2(n):
    name = "Diagonal 2 function"
    sm = lambda i: sp.exp(xi(i)) - xi(i) / i
    x0 = (1 / np.arange(1, n + 1)).reshape((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_3(n):
    name = "Diagonal 3 function"
    sm = lambda i: sp.exp(xi(i)) - i * sp.sin(xi(i))
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_4(n):
    name = "Diagonal 4 function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda i: 0.5 * (xi(2*i-1) + 100*xi(2*i))**2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0)


def diagonal_5(n):
    name = "Diagonal 5 function"
    sm = lambda i: sp.log(sp.exp(xi(i)) + sp.exp(-xi(i)))
    x0 = np.ones((n, 1)) * 1.1
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_6(n):
    name = "Diagonal 6 function"
    sm = lambda i: sp.exp(xi(i)) + (1 - xi(i))
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_7(n):
    name = "Diagonal 7 function"
    sm = lambda i: sp.exp(xi(i)) - 2 * xi(i) - xi(i) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_8(n):
    name = "Diagonal 8 function"
    sm = lambda i: xi(i) * sp.exp(xi(i)) - 2 * xi(i) - xi(i) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def diagonal_9(n):
    name = "Diagonal 9 function"
    sm = lambda i: (sp.exp(xi(i)) - i * xi(i)) + 10000 * xi(n) ** 2
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def ext_block_diagonal(n):
    name = "Extended BD1 function (Block Diagonal)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda i: (xi(2*i-1)**2 + xi(2*i)**2 - 2) ** 2
    sm_2 = lambda i: (sp.exp(xi(2*i-1)-1) - xi(2*i)) ** 2
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1)) * 0.1
    return create_test_function(name, n, sm, x0)


def ext_tridiagonal_1(n):
    name = "Extended Tridiagonal 1 function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda i: (xi(2*i-1) + xi(2*i) - 3) ** 2
    sm_2 = lambda i: (xi(2*i-1) - xi(2*i) + 1) ** 4
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0)


def gen_tridiagonal_1(n):
    name = "Generalized Tridiagonal 1 function"
    sm_1 = lambda i: (xi(i) + xi(i+1) - 3) ** 2
    sm_2 = lambda i: (xi(i) - xi(i+1) + 1) ** 4
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def gen_tridiagonal_2(n):
    name = "Generalized Tridiagonal 2 function"
    f = lambda: ((5 - 3*xi(1) - xi(2)**2) * xi(1) - 3*xi(2) + 1) ** 2
    sm_1 = lambda i: ((5 - 3*xi(i) - xi(i)**2) * xi(1) - xi(i-1) - 3*xi(i+1) + 1) ** 2
    sm_2 = lambda: ((5 - 3*xi(n) - xi(n)**2) * xi(n) - xi(n-1) + 1) ** 2
    sm = lambda i: sm_1(i) + sm_2()
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))


def ext_tridiagonal_2(n):
    name = "Extended Tridiagonal 2 function"
    sm_1 = lambda i: (xi(i) * xi(i+1) - 1) ** 2
    sm_2 = lambda i: 0.1 * (xi(i) + 1) * (xi(i+1) + 1)
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def broyden_tridiagonal(n):
    name = "Broyden Tridiagonal function"
    f = lambda: (3 * xi(1) - 2 * xi(1) ** 2) ** 2
    sm_1 = lambda i: (3*xi(i) - 2*xi(i)**2 - xi(i-1) - 2*xi(i+1) + 1) ** 2
    sm_2 = lambda: (3*xi(n) - 2*xi(n)**2 - xi(n-1) + 1) ** 2
    sm = lambda i: sm_1(i) + sm_2()
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))


def perturbed_tridiagonal_quadratic(n):
    name = "Perturbed Tridiagonal Quadratic function"
    f = lambda: xi(1) ** 2
    sm = lambda i: i * xi(i) ** 2 + (xi(i-1) + xi(i) + xi(i+1)) ** 2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))
