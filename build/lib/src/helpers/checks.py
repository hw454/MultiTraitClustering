import pandas as pd

def str_check(var, var_name):
    """Raise Type Error if var is not a string
    
    Args:
        var: The variable to be checked
        var_name: The name of the variable for the error string.

    Return: None
    """
    
    if not isinstance(var,str):
        var_type = type(var)
        error_string = """The input {name} should be a string not {var_type}""" .format(name = var_name, var_type = str(var_type))
        raise TypeError(error_string)
    return

def int_check(var, var_name):
    """Raise Type Error if var is not a int
    
    Args:
        var: The variable to be checked
        var_name: The name of the variable for the error string.

    Return: None
    """
    
    if not isinstance(var, int):
        var_type = type(var)
        error_string = """The input {name} should be a int not {var_type}""" .format(name = var_name, var_type = str(var_type))
        raise TypeError(error_string)
    return

def float_check(var, var_name):
    """Raise Type Error if var is not a float
    
    Args:
        var: The variable to be checked
        var_name: The name of the variable for the error string.

    Return: None
    """
    
    if not isinstance(var, float):
        var_type = type(var)
        error_string = """The input {name} should be a float not {var_type}""" .format(name = var_name, var_type = str(var_type))
        raise TypeError(error_string)
    return

def df_check(var, var_name):
    """Raise Type Error if var is not a Dataframe
    
    Args:
        var: The variable to be checked
        var_name: The name of the variable for the error string.

    Return: None
    """
    
    if not isinstance(var, pd.DataFrame):
        var_type = type(var)
        error_string = """The input {name} should be a dataframe not {var_type}""" .format(name = var_name, var_type = str(var_type))
        raise TypeError(error_string)
    return