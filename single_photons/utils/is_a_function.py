def is_a_function(maybe_a_function):
    """
    Auxiliar internal function checking if a given variable is a lambda function
    """
    return callable(maybe_a_function)                   # OLD: isinstance(obj, types.LambdaType) and obj.__name__ == "<lambda>"