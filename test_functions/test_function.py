import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from .support_funcs import default_range_2
from .exceptions import MinDimensionError


"""
Test function types are:
  1.Unimodal function.
  2.Functions with significant null-space effects.
  3.Essentially unimodal functions.
  4.Functions with a huge number of significant local optima.
  5.Functions with a small number of significant local optima
  6.Functions whose global structure provides no useful information about its optima.
"""
#TODO put each test function in it's type python file


class TestFunction:
    """
    TestFunction class describe a test functions for unconstrained optimization.
    It based on sympy function representation.
    So it consist of function variables, function form, default x0 and minimum dimension.
    You can lamdify
    """
    def __init__(self, name, function, x0, min_dimesion):
        self.variables = function.free_symbols
        self.dimension = len(self.variables)
        if self.dimension < min_dimesion:
            raise MinDimensionError(min_dimesion, self.dimension)
        self.name = name
        self.sympy_function = function
        self.x0 = x0

    def lambdify(self):
        f = sp.lambdify(self.variables, self.sympy_function)
        return lambda x: f(*x)

    def plot_surface_3d(self, x1, x2):
        if self.dimension == 2:
            fig = plt.figure(self.name)
            ax = fig.gca(projection='3d')
            X = x1
            Y = x2
            func = self.lambdify()
            X, Y = np.meshgrid(X, Y)
            Z = np.array([func((x, y)) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
            ax.plot_surface(X, Y, Z)
        else:
            print("The dimesion is too high for plotting")

    @property
    def get_name(self):
        return self.name

    @property
    def get_variables(self):
        return self.variables

    @property
    def get_function(self):
        return self.sympy_function

    @property
    def get_dimension(self):
        return self.dimension

    @property
    def get_x0(self):
        return self.x0

    def __str__(self):
        result = "Test function name: {self.name}\n" \
                 "Function dimension: {self.dimension}\n" \
                 "Function form: {self.sympy_function}\n" \
                 "Default x0: \n{self.x0}"
        return result.format(self=self)


def xi(i):
    return sp.Symbol('x%i' % i)


def summation(series_member, series_range):
    return sum([series_member(i) for i in series_range])


def create_test_function(name, n, series_member, x0,
                         first=None, range_func=default_range_2, limits=None,
                         min_dimesion=2):
    if limits:
        series_range = range_func(*limits)
    else:
        series_range = range_func(n)
    func = summation(series_member, series_range)
    if first:
        func += first()
    return TestFunction(name, func, x0, min_dimesion)




