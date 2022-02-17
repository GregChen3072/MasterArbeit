from turtle import delay
from dask import delayed


def my_squared_function(x):
    return x**2


delayed_square_function = delayed(my_squared_function)(4)

print(delayed_square_function.compute())
