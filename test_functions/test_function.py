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
    def __init__(self, name, variables, function, x0, min_dimesion):
        if len(variables) < min_dimesion:
            raise MinDimensionError(len(variables))
        self.name = name
        self.variables = variables
        self.sympy_function = function
        self.dimension = len(variables)
        self.x0 = x0

    def lambdify(self):
        return sp.lambdify(self.variables, self.sympy_function)

    def plot_surface_3d(self, x1, x2):
        if self.dimension == 2:
            fig = plt.figure(self.name)
            ax = fig.gca(projection='3d')
            X = x1
            Y = x2
            func = self.lambdify()
            X, Y = np.meshgrid(X, Y)
            Z = np.array([func(x, y) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
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


class FunctionSeries:
    """
    FunctionSeries is class-functor. Often test function fo UO is represented as a series.
    This class create a sympy function based on the member of series and  it's range
    """
    class ListFrom1(list):
        """ListFrom1 is just a default python list but it's index must be in range [1, n]"""
        def __init__(self, lst):
            super().__init__(lst)

        def __getitem__(self, item):
            if item < 1 or item > super().__len__():
                raise IndexError("list index out of range")
            return super().__getitem__(item - 1)

    def __init__(self, member, n, series_range):
        self.member = member
        self.n = n
        self.range = series_range

    def __call__(self, *args, **kwargs):
        x = self.ListFrom1(sp.symbols(' '.join(['x' + str(i) for i in range(1, self.n + 1)])))
        if isinstance(x, sp.Symbol):
            x = [x]
        func = sum([self.member(x, i) for i in self.range])
        return x, func


def create_test_function(name, n, series_member, x0,
                         first=None, range_func=default_range_2, limits=None,
                         min_dimesion=2):
    if limits:
        series_range = range_func(*limits)
    else:
        series_range = range_func(n)
    variables, func = FunctionSeries(series_member, n, series_range)()
    if first:
        func += first(variables)
    return TestFunction(name, variables, func, x0, min_dimesion)




