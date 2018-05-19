from .test_function import *
from .support_funcs import *
from .exceptions import *


def FLETCBV3(n):
    name = "FLETCBV3 function (CUTE)"
    print(name)
    p, h = 1e-8, 1 / (n + 1)
    # move last n-member of second series to f and and create one series from 1 to n-1
    f = lambda x: 0.5*p*(x[1] + x[n])**2 - ((h**2 + 2) * x[n] - sp.cos(x[n])) * p / h**2
    sm_1 = lambda x, i: 0.5*p*(x[i] - x[i+1])
    sm_2 = lambda x, i: -((h**2 + 2) * x[i] - sp.cos(x[i])) * p / h**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = (np.arange(1, n + 1) * h).reshape((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def FLETCHCR(n):
    name = "FLETCHCR function (CUTE)"
    sm = lambda x, i: 100 * (x[i+1] - x[i] + 1 - x[i]**2)**2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def BDQRTIC(n):
    name = "BDQRTIC function (CUTE)"
    sm_1 = lambda x, i: (-4*x[i] + 3)**2
    sm_2 = lambda x, i: (x[i] + 2*x[i+1]**2 + 3*x[i+2]**2 + 4*x[i+3] + 5*x[n]**2) ** 2
    sm = lambda x, i:  sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range, limits=(1, n - 3))


def TRIDIA(n):
    name = "TRIDIA function (CUTE)"
    alpha, beta, gamma, sigma = 2, 1, 1, 1
    f = lambda x: gamma * (sigma * x[1] - 1) ** 2
    sm = lambda x, i:  i * (alpha * x[i] - beta * x[i-1]) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, first=f,
                                range_func=default_range, limits=(2, n + 1))


def ARGLINB(n):
    name = "ARGLINB function (CUTE)"
    sm = lambda x, i: sum([i*j*x[j] - 1 for j in range(1, n + 1)]) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ARWHEAD(n):
    name = "ARWHEAD function (CUTE)"
    sm = lambda x, i: (-4*x[i] + 3)**2 + (x[i]**2 + x[n]**2) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def NONDIA(n):
    name = "NONDIA function (CUTE)"
    f = lambda x: (x[1] - 1) ** 2
    sm = lambda x, i: 100 * (x[i] - x[i-1]**2)**2
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n + 1))


def NONDQUAR(n):
    name = "NONDQUAR function (CUTE)"
    f = lambda x: (x[1] - x[2]) ** 2
    sm_1 = lambda x, i: (x[i] + x[i+1] + x[n]) ** 4
    sm_2 = lambda x: (x[n-1] + x[n]) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x)
    x0 = construct_x0([[1.0], [-1.0]], n)
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n - 1))


def DQDRTIC(n):
    name = "DQDRTIC function (CUTE)"
    c, d = 100, 100
    sm = lambda x, i: x[i] + c*x[i+1]**2 + d*x[i+2]**2
    x0 = np.ones((n, 1)) * 3.0
    return create_test_function(name, n, sm, x0,
                                range_func=default_range, limits=(2, n - 1), min_dimesion=3)


def EG2(n):
    name = "EG2 function (CUTE)"
    sm = lambda x, i: sp.sin(x[1] + x[i]**2 - 1) + 0.5 * sp.sin(x[n]**2)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def CURLY20(n):
    def q(x, i, k=20):
        if i <= n - k:
            return sum([x[i] for i in range(1, i + k + 1)])
        else:
            return sum([x[i] for i in range(1, n + 1)])
    name = "CURLY20 function (CUTE)"
    sm = lambda x, i: q(x, i)**2 - 20*q(x, i)**2 - 0.1*q(x,i)
    x0 = np.ones((n, 1)) * 0.001 / (n + 1)
    return create_test_function(name, n, sm, x0, range_func=default_range_1)



def LIARWHD_1(n):
    name = "LIARWHD1 function (CUTE)"
    sm = lambda x, i: 4*(-x[1] + x[i]**2)**2 + (x[i] - 1)**2
    x0 = np.ones((n, 1)) * 4.0
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def POWER(n):
    name = "POWER function (CUTE)"
    sm = lambda x, i: (i * x[i]) ** 2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def ENGVAL1(n):
    name = "ENGVAL1 function (CUTE)"
    sm = lambda x, i: (x[i]**2 + x[i+1]**2)**2 + (-4*x[i] + 3)
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def CRAGGLVY(n):
    name = "CRAGGLVY function (CUTE)"
    if n % 2 or n < 4:
        raise DimensionError(name, 2, n, 4)
    sm_1 = lambda x, i: (sp.exp(x[2*i-1] - x[2*i]))**4 + 100*(x[2*i] - x[2*i-1])**6
    sm_2 = lambda x, i: (sp.tan(x[2*i+1]+x[2*i+2]) + x[2*i+1] - x[2*i+2])**4
    sm_3 = lambda x, i: x[2*i-1]**8 + (x[2*i+2]-1)**2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i) + sm_3(x, i)
    x0 = np.ones((n, 1)) * 2.0
    x0[0][0] = 1.0
    return create_test_function(name, n, sm, x0, range_func=default_range, limits=(1, n // 2))


def EDENSCH(n):
    name = "EDENSCH function (CUTE)"
    f = lambda x: 16
    sm = lambda x, i: (x[i] - 2)**4 + (x[i]*x[i+1] - 2*x[i+1])**2 +(x[i+1] + 1)**2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def INDEF(n):
    name = "INDEF function (CUTE)"
    f = lambda x: x[1] + x[n]
    sm = lambda x, i: x[i] + 0.5 * sp.cos(2*x[i] - x[n] - x[1])
    x0 = (np.arange(1, n + 1) * 1 / (n + 1)).reshape((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range, limits=(2, n))


def CUBE(n):
    name = "CUBE function (CUTE)"
    f = lambda x: (x[1] - 1)**2
    sm = lambda x, i: 100*(x[i] - x[i-1]**3)**2
    x0 = construct_x0([[-1.2], [1.0]], n)
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n + 1))


def EXPLIN1(n):
    name = "EXPLIN1 function (CUTE)"
    f = lambda x: -10 * n * x[n]
    sm = lambda x, i: sp.exp(0.1*x[i]*x[i+1]) - 10*(i*x[i])
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def EXPLIN2(n):
    name = "EXPLIN2 function (CUTE)"
    f = lambda x: -10 * n * x[n]
    sm = lambda x, i: sp.exp(i*x[i]*x[i+1] / (10*n)) - 10*(i*x[i])
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def ARGLINC(n):
    name = "ARGLINC function (CUTE)"
    f = lambda x: 2
    sm = lambda x, i: sum([j*x[j]*(i-1) - 1 for j in range(2, n)])**2
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n))


def BDEXP(n):
    name = "BDEXP function (CUTE)"
    sm = lambda x, i: (x[i] + x[i+1]) * sp.exp(-x[i+2]*(x[i] + x[i+1]))
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range, limits=(1, n - 1))


def HARKERP2(n):
    name = "HARKERP2 function (CUTE)"
    f1 = lambda x, j: sum([x[j] for j in range(j, n + 1)]) ** 2
    f2 = lambda x: sum([f1(x, j) for j in range(2, n + 1)])
    f = lambda x: f1(x, 1) + 2 * f2(x)
    sm = lambda x, i: -(x[i] + 0.5*x[i]**2)**2
    x0 = np.arange(1, n + 1).reshape((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def GENHUMPS(n):
    name = "GENHUMPS function (CUTE)"
    sm_1 = lambda x, i: (sp.sin(2*x[i])**2) * sp.sin(2*x[i+1])**2
    sm_2 = lambda x, i: 0.05*(x[i]**2 + x[i+1]**2)
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1)) * -506.2
    x0[0][0] = 506.0
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def MCCORMCK(n):
    name = "MCCORMCK function (CUTE)"
    sm_1 = lambda x, i: -1.5*x[i] + 2.5*x[i+1] + 1
    sm_2 = lambda x, i: (x[i] - x[i+1])**2 + sp.sin(x[i] + x[i+1])
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def NONSCOMP(n):
    name = "NONSCOMP function (CUTE)"
    f = lambda x: (x[1] - 1) ** 2
    sm = lambda x, i: 4 * (x[i] - x[i-1]**2)**2
    x0 = np.ones((n, 1)) * 3.0
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n + 1))


def VARDIM(n):
    name = "VARDIM function (CUTE)"
    sm_f = lambda x, i: i* x[i] - n * (n + 1) / 2
    ff = lambda x: sum([sm_f(x, j) for j in range(1, n + 1)])
    f = lambda x: ff(x) ** 2 + ff(x) ** 4
    sm = lambda x, i: (x[i] - 1) ** 2
    x0 = (np.ones(n) - np.arange(1, n + 1) / n).reshape((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)


def QUARTC(n):
    name = "QUARTC function (CUTE)"
    sm = lambda x, i: (x[i] - 1) ** 4
    x0 = np.ones((n, 1)) * 2.0
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def SINQUAD(n):
    name = "SINQUAD function (CUTE)"
    f = lambda x: (x[1] - 1) ** 4
    sm_1 = lambda x, i: (sp.sin(x[i] - x[n]) - x[1] + x[i]**2)**2
    sm_2 = lambda x: (x[n]**2 - x[1]**2) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x)
    x0 = np.ones((n, 1)) * 0.1
    return create_test_function(name, n, sm, x0,
                                first=f, range_func=default_range, limits=(2, n))


def ext_DENSCHNB(n):
    name = "Extended DENSCHNB function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (x[2*i-1] - 2)**2 +(x[2*i-1] - 2) * x[2*i] ** 2
    sm_2 = lambda x, i: (x[2*i] + 1) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0)


def ext_DENSCHNF(n):
    name = "Extended DENSCHNF function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (2*(x[2*i-1] + x[2*i])**2 + (x[2*i-1] - x[2*i])**2 - 8) ** 2
    sm_2 = lambda x, i: (5*x[2*i-1]**2 + (x[2*i] - 3)**2 - 9) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[0.0], [2.0]], n)
    return create_test_function(name, n, sm, x0)


def LIARWHD_2(n):
    name = "LIARWHD2 function (CUTE)"
    sm = lambda x, i: 4*(x[i]**2 - x[i]) ** 2 + (x[i] - 1) ** 2
    x0 = np.ones((n, 1)) * 4.0
    return create_test_function(name, n, sm, x0, range_func=default_range_1)


def DIXON3DQ(n):
    name = "DIXON3DQ function (CUTE)"
    f = lambda x: (x[1] - 1) ** 2
    sm = lambda x, i: (x[i] - x[i+1]) ** 2 + (x[n] - 1) ** 2
    x0 = np.ones((n, 1)) * -1.0
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def COSINE(n):
    name = "COSINE function (CUTE)"
    sm = lambda x, i: sp.cos(-0.5*x[i+1] + x[i]**2)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def SINE(n):
    name = "SINE function (CUTE)"
    sm = lambda x, i: sp.sin(-0.5*x[i+1] + x[i]**2)
    x0 = np.ones((n, 1))
    return create_test_function(name, n, sm, x0, range_func=default_range_3)


def BIGGSB1(n):
    name = "BIGGSB1 function (CUTE)"
    f = lambda x: (x[1] - 1) ** 2
    sm = lambda x, i: (x[i+1] - x[i]) ** 2 + (1 - x[n]) ** 2
    x0 = np.zeros((n, 1))
    return create_test_function(name, n, sm, x0, first=f, range_func=default_range_3)


def SINCOS(n):
    name = "SINCOS function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm_1 = lambda x, i: (x[2*i-1]**2 + x[2*i]**2 + x[2*i-1]*x[2*i]) ** 2
    sm_2 = lambda x, i: sp.sin(x[2*i-1]) ** 2 + sp.cos(x[2*i]) ** 2
    sm = lambda x, i: sm_1(x, i) + sm_2(x, i)
    x0 = construct_x0([[3.0], [0.1]], n)
    return create_test_function(name, n, sm, x0)


def HIMMELBG(n):
    name = "HIMMELBG function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda x, i: (2*x[2*i-1]**2 + 3*x[2*i]**2) * sp.exp(-x[2*i-1] - x[2*i])
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0)


def HIMMELH(n):
    name = "HIMMELH function (CUTE)"
    if n % 2:
        raise DimensionError(name, 2, n)
    sm = lambda x, i: -3*x[2*i-1] - 2*x[2*i] + 2 + x[2*i-1] ** 2 + x[2*i] ** 2
    x0 = np.ones((n, 1)) * 1.5
    return create_test_function(name, n, sm, x0)