from .test_function import *
from .support_funcs import *
from .exceptions import *


def perturbed_quadratic(n):
    name = "Perturbed Quadratic function"
    f = lambda x: 0.01 * sum([x[i] for i in range(1, n + 1)]) ** 2
    sm = lambda x, i: i * x[i] ** 2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def quadratic_1(n):
    name = "Quadratic QF1 function"
    sm = lambda x, i: 0.5 * (i * x[i] ** 2 - x[n])
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def quadratic_2(n):
    name = "Quadratic QF2 function"
    sm = lambda x, i: 0.5 * (i * (x[i] ** 2 - 1) ** 2 - x[n])
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ext_quadratic_penalty_1(n):
    name = "Extended quadratic penalty QP1 function"
    f = lambda x: (sum([x[i] ** 2 for i in range(1, n + 1)]) - 0.5) ** 2
    sm = lambda x, i: (x[i] ** 2 - 2) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def ext_quadratic_penalty_2(n):
    name = "Extended quadratic penalty QP2 function"
    f = lambda x: (sum([x[i] ** 2 for i in range(1, n + 1)]) - 100) ** 2
    sm = lambda x, i: (x[i] ** 2 - sp.sin(x[i])) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def ext_quadratic_exponential(n):
    name = "Extended quadratic exponential EP1 function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (sp.exp(x[2*i-1] - x[2*i]) - 5) ** 2
    sm_2 = lambda x, i: ((x[2*i-1] - x[2*i])**2) * (x[2*i-1] - x[2*i] - 11)**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0)


def partial_perturbed_quadratic(n):
    name = "Partial Perturbed Quadratic function"
    f = lambda x: x[1] ** 2
    sm = lambda x, i: i * x[i] ** 2 + 0.01 * sum([x[j] for j in range(1, i + 1)])**2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def almost_perturbed_quadratic(n):
    name = "Almost Perturbed Quadratic function"
    sm = lambda x, i: i * x[i] ** 2 + 0.01 * (x[1] + x[n]) ** 2
    x0 = np.ones((n, 1)) * 0.5
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def gen_quadratic(n):
    name = "Generalized Quadratic function"
    sm = lambda x, i: x[i] ** 2 + (x[i+1] + x[i] ** 2) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)