from .test_function import *
from .support_funcs import *

table_DIXMAAN = dict()
table_DIXMAAN['A'] = (1, 0, 0.125, 0.125, 0, 0, 0, 0)
table_DIXMAAN['B'] = (1, 0.0625, 0.0625, 0.0625, 0, 0, 0, 1)
table_DIXMAAN['C'] = (1, 0.125, 0.125, 0.125, 0, 0, 0, 0)
table_DIXMAAN['D'] = (1, 0.26, 0.26, 0.26, 0, 0, 0, 0)
table_DIXMAAN['E'] = (1, 0, 0.125, 0.125, 1, 0, 0, 1)
table_DIXMAAN['F'] = (1, 0.0625, 0.0625, 0.0625, 1, 0, 0, 1)
table_DIXMAAN['G'] = (1, 0.125, 0.125, 0.125, 1, 0, 0, 1)
table_DIXMAAN['H'] = (1, 0.26, 0.26, 0.26, 1, 0, 0, 1)
table_DIXMAAN['I'] = (1, 0, 0.125, 0.125, 2, 0, 0, 2)
table_DIXMAAN['J'] = (1, 0.0625, 0.0625, 0.0625, 2, 0, 0, 2)
table_DIXMAAN['K'] = (1, 0.125, 0.125, 0.125, 2, 0, 0, 2)
table_DIXMAAN['L'] = (1, 0.26, 0.26, 0.26, 2, 0, 0, 2)


def DIXMAAN(type):
    def DIXMAAN_(n):
        name = "DIXMAAN%c function (CUTE)" % type
        alpha, beta, gamma, sigma, k1, k2, k3, k4 = table_DIXMAAN[type]
        m = n // 3
        sm = lambda x, i: alpha * x[i] ** 2 *(i / n) ** k1
        sm2 = lambda x, i: beta * x[i] ** 2 * (x[i+1] + x[i+1]**2) * (i / n) ** k2
        sm3 = lambda x, i: gamma * x[i]**2 * x[i+m] ** 4 * (i / n) ** k3
        sm4 = lambda x, i: sigma * x[i] * x[i+2*m] * (i / n) ** k4
        f_1 = lambda x: sum([sm2(x, i) for i in range(1, n)])
        f_2 = lambda x: sum([sm3(x, i) for i in range(1, 2 * m + 1)])
        f_3 = lambda x: sum([sm4(x, i) for i in range(1, m + 1)])
        f = lambda x: 1 + f_1(x) + f_2(x) + f_3(x)
        x0 = np.ones((n, 1)) * 2.0
        return create_test_function(name, n, sm, x0, first=f, range_func=default_range_1)
    DIXMAAN_.__name__ += type
    return DIXMAAN_