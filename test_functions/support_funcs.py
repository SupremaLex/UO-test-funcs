import numpy as np


def default_range(start, stop):
    """[start, start + 1, ..., stop -1]"""
    return range(start, stop)


def default_range_1(n):
    """[1, 2, ..., n]"""
    return range(1, n + 1)


def default_range_2(n):
    """[1, 2, ..., n // 2]"""
    return range(1, n // 2 + 1)


def default_range_3(n):
    """[1, 2, ..., n - 1]"""
    return range(1, n)


def construct_x0(base, n):
    return np.array(base * (n // len(base)))
