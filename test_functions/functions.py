from .test_function import *
from .support_funcs import *
from .exceptions import *


def ext_freudenstein_and_roth(n):
    name = "Extended Freudenstein & Roth function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (-13 + x[2*i-1] + ((5 - x[2*i])*x[2*i] - 2)*x[2*i])**2
    sm_2 = lambda x, i: (-29 + x[2*i-1] + ((x[2*i] + 1)*x[2*i] - 14)*x[2*i])**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[0.5], [-2.0]], n)
    return create_test_function(name, n, sm, x0)


def ext_trigonometric(n):
    name = "Extended Trigonometric function"
    sm_1 = lambda x, i: n - sum([sp.cos(x[j]) for j in range(1, n + 1)])
    sm_2 = lambda x, i: i*(1 - sp.cos(x[i])) - sp.sin(x[i])
    sm = lambda x, i: (sm_1(x, i) + sm_2(x, i))**2
    x0 = np.ones((n, 1)) * 0.2
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ext_rosenbrock(n):
    name = "Extended Rosenbrock function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda x, i: 100*(x[2*i]-x[2*i-1]**2)**2 + (1 - x[2*i-1])**2
    x0 = construct_x0([[1], [-1.2]], n)
    return create_test_function(name, n, sm, x0)


def gen_rosenbrock(n):
    name = "Generalized Rosenbrock function"
    sm = lambda x, i: 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    x0 = construct_x0([[1], [-1.2]], n)
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def ext_white_and_holst(n):
    name = "Extended White & Holst function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: 100 * (x[2 * i] - x[2 * i - 1] ** 3) ** 2
    sm_2 = lambda x, i: (1 - x[2 * i - 1]) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[1], [-1.2]], n)
    return create_test_function(name, n, sm, x0)


def ext_beale(n):
    name = "Extended Beale function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (1.5 - x[2*i-1]*(1-x[2*i]))**2
    sm_2 = lambda x, i: (2.25 - x[2*i-1]*(1-x[2*i]**2))**2
    sm_3 = lambda x, i: (2.625 - x[2 * i - 1] * (1 - x[2 * i]**3)) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i) + sm_3(x, i)
    x0 = construct_x0([[1], [0.8]], n)
    return create_test_function(name, n, sm, x0)


def ext_penalty(n):
    name = "Extended Penalty function"
    sm_1 = lambda x, i: (x[i] - 1)**2
    sm_2 = lambda x: (sum([x[j]**2 for j in range(1, n + 1)]) - 0.25)**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x)
    x0 = np.arange(1, n + 1).reshape((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def raydan_1(n):
    name = "Raydan 1 function"
    sm = lambda x, i: i * (sp.exp(x[i]) - x[i]) / 10
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def raydan_2(n):
    name = "Raydan 2 function"
    sm = lambda x, i: sp.exp(x[i]) - x[i]
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def hager(n):
    name = "Hager function"
    sm = lambda x, i: sp.exp(x[i]) - sp.sqrt(i) * x[i]
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ext_three_exponential_terms(n):
    name = "Extended TET function : (Three exponential terms)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: sp.exp(x[2*i-1] + 3*x[2*i] - 0.1)
    sm_2 = lambda x, i: sp.exp(x[2*i-1] - 3*x[2*i] - 0.1)
    sm_3 = lambda x, i: sp.exp(-x[2*i-1] - 0.1)
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i) + sm_3(x, i)
    x0 = np.ones((n, 1)) * 0.1
    return create_test_function(name, n, sm, x0)


def ext_himmelblau(n):
    name = "Extended Himmelblau function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (x[2*i-1]**2 + x[2*i] - 11)**2
    sm_2 = lambda x, i: (x[2*i-1] + x[2*i]**2 - 7)**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0)


def gen_white_and_holst(n):
    name = "Generalized White & Holst function"
    sm = lambda x, i: 100 * (x[i+1] - x[i]**3)**2 + (1 - x[i])**2
    x0 = construct_x0([[1], [-1.2]], n)
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def gen_PCS1(n):
    name = "Generalized PSC1 function"
    sm_1 = lambda x, i: (x[i]**2 + x[i+1]**2 + x[i]*x[i+1])**2
    sm_2 = lambda x, i: sp.sin(x[i])**2 + sp.cos(x[i])**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[3.0], [0.1]], n)
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def ext_PSC1(n):
    name = "Extended PSC1 function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (x[2*i-1]**2 + x[2*i]**2 + x[2*i-1]*x[2*i])**2
    sm_2 = lambda x, i: sp.sin(x[2*i-1])**2 + sp.cos(x[2*i])**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[3.0], [0.1]], n)
    return create_test_function(name, n, sm, x0)


def ext_powell(n):
    name = "Extended Powell function"
    if n % 4:
        raise DimensionError(name, 4, n)
    sm_1 = lambda x, i: (x[4*i-3] + 10*x[4*i-2])**2 + 5*(x[4*i-1] - x[4*i])**2
    sm_2 = lambda x, i: (x[4*i-2] - 2*x[4*i-1])**4 + 10*(x[4*i - 3] - x[4*i])**4
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[-3], [-1], [0], [1]], n)
    return create_test_function(name, n, sm, x0,
                                range_func=default_range, limits=(1, n // 4 + 1), min_dimesion=4)


def ext_maratos(n):
    name = "Extended Maratos function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda x, i: x[2*i-1] + 100*(x[2*i-1]**2 + x[2*i]**2 - 1)**2
    x0 = construct_x0([[1.1], [0.1]], n)
    return create_test_function(name, n, sm, x0)


def ext_cliff(n):
    name = "Extended Cliff function"
    if n % 2:
        raise DimensionError(name, 2, n, 4)
    sm_1 = lambda x, i: ((x[2*i-1] - 3) / 100)**2 - (x[2*i-1] - x[2*i])
    sm_2 = lambda x, i: sp.exp(20*(x[2*i-1] - x[2*i]))
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[0.0], [-1.0]], n)
    return create_test_function(name, n, sm, x0)


def ext_wood(n):
    name = "Extended Wood function"
    if n % 4:
        raise DimensionError(name, 4, n, 4)
    sm_1 = lambda x, i: 100 * (x[4*i-3]**2 - x[4*i-2])**2 + (x[4*i-3] - 1)**2
    sm_2 = lambda x, i: 90 * (x[4*i-2]**2 + x[4*i])**2 + (1 - x[4*i-1])**2
    sm_3 = lambda x, i: 10.1 * ((x[4*i-2] - 1)**2 + (x[4*i] - 1)**2)
    sm_4 = lambda x, i: 19.8 * (x[4*i-2] - 1)*(x[4*i] - 1)
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i) + sm_3(x, i) + sm_4(x, i)
    x0 = construct_x0([[-3.0], [-1.0], [-3.0], [-1.0]], n)
    return create_test_function(name, n, sm, x0,
                                range_func=default_range, limits=(1, n // 4 + 1), min_dimesion=4)


def ext_hiebert(n):
    name = "Extended Hiebert function"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda x, i: (x[2*i-1] - 10)**2 + (x[2*i-1]*x[2*i] - 50000)**2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0)


def staircase_1(n):
    name = "Staircase 1 function"
    sm = lambda x, i: sum([x[j] for j in range(1, i + 1)])**2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def staircase_2(n):
    name = "Staircase 2 function"
    sm = lambda x, i: (sum([x[j] for j in range(1, i + 1)]) - i)**2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)

