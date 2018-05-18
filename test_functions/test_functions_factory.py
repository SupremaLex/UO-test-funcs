from .test_functions_register import test_function_register


class TestFunctionFactory:
    @staticmethod
    def create(function_name, function_dimesion):
        return test_function_register[function_name](function_dimesion)

    @property
    def get_all_functions_names(self):
        return list(test_function_register.keys())