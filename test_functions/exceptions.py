class TestFunctionsError(Exception):
    def __init__(self, data, message):
        super().__init__()
        self.data = data
        self.message = message

    def __repr__(self):
        super().__str__()

    def __str__(self):
        return str(self.message) + ': ' + str(self.data)


class DimensionError(TestFunctionsError):
    def __init__(self, func_name, number, dimension, min_dim=2):
        msg = "The %s dimension must be multiple of %i and be greater or equal to %i. Dimension"
        msg = msg % (func_name, number, min_dim)
        super().__init__(dimension, msg)


class MinDimensionError(TestFunctionsError):
    def __init__(self, min_dimension, dimesion):
        msg = "The dimesion must be at least %. Dimension" % min_dimension
        super().__init__(dimesion, msg)