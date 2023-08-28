import numba as nb
from math import acos
from time import time


@nb.jit(nopython=True)
def myfunc():
    s = 0
    for i in range(10000000):
        s += acos(0.5)
    return s


@nb.jit(nopython=True)
def myfunc2():
    return myfunc()


tic = time()
x = myfunc2()  # 10471975.511390356
toc = time()
print(toc - tic)
