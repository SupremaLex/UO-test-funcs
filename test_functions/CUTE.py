from .test_function import *
from .support_funcs import *
from .exceptions import *


def FLETCBV3(n):
    name = "FLETCBV3 function (CUTE)"
    print(name)
    p, h = 1e-8, 1 / (n + 1)
    # move last n-member of second series to f and and create one series from 1 to n-1
    f = lambda: 0.5*p*(xi(1) + xi(n))**2 - ((h**2 + 2) * xi(n) - sp.cos(xi(n))) * p / h**2
    sm_1 = lambda i: 0.5*p*(xi(i) - xi(i+1))
    sm_2 = lambda i: -((h**2 + 2) * xi(i) - sp.cos(xi(i))) * p / h**2
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = (np.arange(1, n + 1) * h).reshape((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def FLETCHCR(n):
    name = "FLETCHCR function (CUTE)"
    sm = lambda i: 100 * (xi(i+1) - xi(i) + 1 - xi(i)**2)**2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def BDQRTIC(n):
    name = "BDQRTIC function (CUTE)"
    sm_1 = lambda i: (-4*xi(i) + 3)**2
    sm_2 = lambda i: (xi(i) + 2*xi(i+1)**2 + 3*xi(i+2)**2 + 4*xi(i+3) + 5*xi(n)**2) ** 2
    sm = lambda i:  sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range, limits=(1, n - 3))


def TRIDIA(n):
    name = "TRIDIA function (CUTE)"
    alpha, beta, gamma, sigma = 2, 1, 1, 1
    f = lambda: gamma * (sigma * xi(1) - 1) ** 2
    sm = lambda i:  i * (alpha * xi(i) - beta * xi(i-1)) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, first=f,
                                range_func=default_range, limits=(2, n + 1))


def ARGLINB(n):
    name = "ARGLINB function (CUTE)"
    sm = lambda i: sum([i*j*xi(j) - 1 for j in range(1, n + 1)]) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ARWHEAD(n):
    name = "ARWHEAD function (CUTE)"
    sm = lambda i: (-4*xi(i) + 3)**2 + (xi(i)**2 + xi(n)**2) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def NONDIA(n):
    name = "NONDIA function (CUTE)"
    f = lambda: (xi(1) - 1) ** 2
    sm = lambda i: 100 * (xi(i) - xi(i-1)**2)**2
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n + 1))


def NONDQUAR(n):
    name = "NONDQUAR function (CUTE)"
    f = lambda: (xi(1) - xi(2)) ** 2
    sm_1 = lambda i: (xi(i) + xi(i+1) + xi(n)) ** 4
    sm_2 = lambda: (xi(n-1) + xi(n)) ** 2
    sm = lambda i: sm_1(i) + sm_2()
    x0 = construct_x0([[1.0], [-1.0]], n)
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n - 1))


def DQDRTIC(n):
    name = "DQDRTIC function (CUTE)"
    c, d = 100, 100
    sm = lambda i: xi(i) + c*xi(i+1)**2 + d*xi(i+2)**2
    x0 = np.ones((n, 1)) * 3.0
    return create_test_function(name, n, sm, x0,
                                range_func=default_range, limits=(2, n - 1), min_dimesion=3)


def EG2(n):
    name = "EG2 function (CUTE)"
    sm = lambda i: sp.sin(xi(1) + xi(i)**2 - 1) + 0.5 * sp.sin(xi(n)**2)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def CURLY20(n):
    def q(i, k=20):
        if i <= n - k:
            return sum([xi(i) for i in range(1, i + k + 1)])
        else:
            return sum([xi(i) for i in range(1, n + 1)])
    name = "CURLY20 function (CUTE)"
    sm = lambda i: q(i)**2 - 20*q(i)**2 - 0.1*q(i)
    x0 = np.ones((n, 1)) * 0.001 / (n + 1)
    return create_test_function(name, n, sm, x0, range_func=default_range_1)



def LIARWHD_1(n):
    name = "LIARWHD1 function (CUTE)"
    sm = lambda i: 4*(-xi(1) + xi(i)**2)**2 + (xi(i) - 1)**2
    x0 = np.ones((n, 1)) * 4.0
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def POWER(n):
    name = "POWER function (CUTE)"
    sm = lambda i: (i * xi(i)) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ENGVAL1(n):
    name = "ENGVAL1 function (CUTE)"
    sm = lambda i: (xi(i)**2 + xi(i+1)**2)**2 + (-4*xi(i) + 3)
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def CRAGGLVY(n):
    name = "CRAGGLVY function (CUTE)"
    if n % 2 or n < 4:
        raise DimensionError(name, 2, n, 4)
    sm_1 = lambda i: (sp.exp(xi(2*i-1) - xi(2*i)))**4 + 100*(xi(2*i) - xi(2*i-1))**6
    sm_2 = lambda i: (sp.tan(xi(2*i+1)+xi(2*i+2)) + xi(2*i+1) - xi(2*i+2))**4
    sm_3 = lambda i: xi(2*i-1)**8 + (xi(2*i+2)-1)**2
    sm = lambda i: sm_1(i) + sm_2(i) + sm_3(i)
    x0 = np.ones((n, 1)) * 2.0
    x0[0][0] = 1.0
    return create_test_function(name, n, sm, x0, range_func=default_range, limits=(1, n // 2))


def EDENSCH(n):
    name = "EDENSCH function (CUTE)"
    f = lambda: 16
    sm = lambda i: (xi(i) - 2)**4 + (xi(i)*xi(i+1) - 2*xi(i+1))**2 +(xi(i+1) + 1)**2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def INDEF(n):
    name = "INDEF function (CUTE)"
    f = lambda: xi(1) + xi(n)
    sm = lambda i: xi(i) + 0.5 * sp.cos(2*xi(i) - xi(n) - xi(1))
    x0 = (np.arange(1, n + 1) * 1 / (n + 1)).reshape((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))


def CUBE(n):
    name = "CUBE function (CUTE)"
    f = lambda: (xi(1) - 1)**2
    sm = lambda i: 100*(xi(i) - xi(i-1)**3)**2
    x0 = construct_x0([[-1.2], [1.0]], n)
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n + 1))


def EXPLIN1(n):
    name = "EXPLIN1 function (CUTE)"
    f = lambda: -10 * n * xi(n)
    sm = lambda i: sp.exp(0.1*xi(i)*xi(i+1)) - 10*(i*xi(i))
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def EXPLIN2(n):
    name = "EXPLIN2 function (CUTE)"
    f = lambda: -10 * n * xi(n)
    sm = lambda i: sp.exp(i*xi(i)*xi(i+1) / (10*n)) - 10*(i*xi(i))
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def ARGLINC(n):
    name = "ARGLINC function (CUTE)"
    f = lambda: 2
    sm = lambda i: sum([j*xi(j)*(i-1) - 1 for j in range(2, n)])**2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n))


def BDEXP(n):
    name = "BDEXP function (CUTE)"
    sm = lambda i: (xi(i) + xi(i+1)) * sp.exp(-xi(i+2)*(xi(i) + xi(i+1)))
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range, limits=(1, n - 1))


def HARKERP2(n):
    name = "HARKERP2 function (CUTE)"
    f1 = lambda j: sum([xi(j) for j in range(j, n + 1)]) ** 2
    f2 = lambda: sum([f1(j) for j in range(2, n + 1)])
    f = lambda: f1(1) + 2 * f2()
    sm = lambda i: -(xi(i) + 0.5*xi(i)**2)**2
    x0 = np.arange(1, n + 1).reshape((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def GENHUMPS(n):
    name = "GENHUMPS function (CUTE)"
    sm_1 = lambda i: (sp.sin(2*xi(i))**2) * sp.sin(2*xi(i+1))**2
    sm_2 = lambda i: 0.05*(xi(i)**2 + xi(i+1)**2)
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1)) * -506.2
    x0[0][0] = 506.0
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def MCCORMCK(n):
    name = "MCCORMCK function (CUTE)"
    sm_1 = lambda i: -1.5*xi(i) + 2.5*xi(i+1) + 1
    sm_2 = lambda i: (xi(i) - xi(i+1))**2 + sp.sin(xi(i) + xi(i+1))
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def NONSCOMP(n):
    name = "NONSCOMP function (CUTE)"
    f = lambda: (xi(1) - 1) ** 2
    sm = lambda i: 4 * (xi(i) - xi(i-1)**2)**2
    x0 = np.ones((n, 1)) * 3.0
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n + 1))


def VARDIM(n):
    name = "VARDIM function (CUTE)"
    sm_f = lambda i: i* xi(i) - n * (n + 1) / 2
    ff = lambda: sum([sm_f(j) for j in range(1, n + 1)])
    f = lambda: ff() ** 2 + ff() ** 4
    sm = lambda i: (xi(i) - 1) ** 2
    x0 = (np.ones(n) - np.arange(1, n + 1) / n).reshape((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def QUARTC(n):
    name = "QUARTC function (CUTE)"
    sm = lambda i: (xi(i) - 1) ** 4
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def SINQUAD(n):
    name = "SINQUAD function (CUTE)"
    f = lambda: (xi(1) - 1) ** 4
    sm_1 = lambda i: (sp.sin(xi(i) - xi(n)) - xi(1) + xi(i)**2)**2
    sm_2 = lambda: (xi(n)**2 - xi(1)**2) ** 2
    sm = lambda i: sm_1(i) + sm_2()
    x0 = np.ones((n, 1)) * 0.1
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n))


def ext_DENSCHNB(n):
    name = "Extended DENSCHNB function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda i: (xi(2*i-1) - 2)**2 +(xi(2*i-1) - 2) * xi(2*i) ** 2
    sm_2 = lambda i: (xi(2*i) + 1) ** 2
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0)


def ext_DENSCHNF(n):
    name = "Extended DENSCHNF function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda i: (2*(xi(2*i-1) + xi(2*i))**2 + (xi(2*i-1) - xi(2*i))**2 - 8) ** 2
    sm_2 = lambda i: (5*xi(2*i-1)**2 + (xi(2*i) - 3)**2 - 9) ** 2
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = construct_x0([[0.0], [2.0]], n)
    return create_test_function(name, n, sm, x0)


def LIARWHD_2(n):
    name = "LIARWHD2 function (CUTE)"
    sm = lambda i: 4*(xi(i)**2 - xi(i)) ** 2 + (xi(i) - 1) ** 2
    x0 = np.ones((n, 1)) * 4.0
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def DIXON3DQ(n):
    name = "DIXON3DQ function (CUTE)"
    f = lambda: (xi(1) - 1) ** 2
    sm = lambda i: (xi(i) - xi(i+1)) ** 2 + (xi(n) - 1) ** 2
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def COSINE(n):
    name = "COSINE function (CUTE)"
    sm = lambda i: sp.cos(-0.5*xi(i+1) + xi(i)**2)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def SINE(n):
    name = "SINE function (CUTE)"
    sm = lambda i: sp.sin(-0.5*xi(i+1) + xi(i)**2)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def BIGGSB1(n):
    name = "BIGGSB1 function (CUTE)"
    f = lambda: (xi(1) - 1) ** 2
    sm = lambda i: (xi(i+1) - xi(i)) ** 2 + (1 - xi(n)) ** 2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def SINCOS(n):
    name = "SINCOS function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda i: (xi(2*i-1)**2 + xi(2*i)**2 + xi(2*i-1)*xi(2*i)) ** 2
    sm_2 = lambda i: sp.sin(xi(2*i-1)) ** 2 + sp.cos(xi(2*i)) ** 2
    sm = lambda i: sm_1(i) + sm_2(i)
    x0 = construct_x0([[3.0], [0.1]], n)
    return create_test_function(name, n, sm, x0)


def HIMMELBG(n):
    name = "HIMMELBG function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda i: (2*xi(2*i-1)**2 + 3*xi(2*i)**2) * sp.exp(-xi(2*i-1) - xi(2*i))
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0)


def HIMMELH(n):
    name = "HIMMELH function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda i: -3*xi(2*i-1) - 2*xi(2*i) + 2 + xi(2*i-1) ** 2 + xi(2*i) ** 2
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0)