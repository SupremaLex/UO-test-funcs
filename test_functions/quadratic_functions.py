from .test_function import *
from .support_funcs import *
from .exceptions import *


def perturbed_quadratic(n):
    name = "Perturbed Quadratic function"
    f = lambda: 0.01 * sum([xi(i) for i in range(1, n + 1)]) ** 2
    sm = lambda i: i * xi(i) ** 2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def quadratic_1(n):
    name = "Quadratic QF1 function"
    sm = lambda i: 0.5 * (i * xi(i) ** 2 - xi(n))
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def quadratic_2(n):
    name = "Quadratic QF2 function"
    sm = lambda i: 0.5 * (i * (xi(i) ** 2 - 1) ** 2 - xi(n))
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ext_quadratic_penalty_1(n):
    name = "Extended quadratic penalty QP1 function"
    f = lambda: (sum([xi(i) ** 2 for i in range(1, n + 1)]) - 0.5) ** 2
    sm = lambda i: (xi(i) ** 2 - 2) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def ext_quadratic_penalty_2(n):
    name = "Extended quadratic penalty QP2 function"
    f = lambda: (sum([xi(i) ** 2 for i in range(1, n + 1)]) - 100) ** 2
    sm = lambda i: (xi(i) ** 2 - sp.sin(xi(i))) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def ext_quadratic_exponential(n):
    name = "Extended quadratic exponential EP1 function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda i: (sp.exp(xi(2*i-1) - xi(2*i)) - 5) ** 2
    sm_2 = lambda i: ((xi(2*i-1) - xi(2*i))**2) * (xi(2*i-1) - xi(2*i) - 11)**2
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0)


def partial_perturbed_quadratic(n):
    name = "Partial Perturbed Quadratic function"
    f = lambda: xi(1) ** 2
    sm = lambda i: i * xi(i) ** 2 + 0.01 * sum([xi(j) for j in range(1, i + 1)])**2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def almost_perturbed_quadratic(n):
    name = "Almost Perturbed Quadratic function"
    sm = lambda i: i * xi(i) ** 2 + 0.01 * (xi(1) + xi(n)) ** 2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def gen_quadratic(n):
    name = "Generalized Quadratic function"
    sm = lambda i: xi(i) ** 2 + (xi(i+1) + xi(i) ** 2) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)