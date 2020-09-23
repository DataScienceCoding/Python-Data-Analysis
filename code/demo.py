import time
import sys
import numpy as np


def vectorSum(n):
    a = list(range(n))
    b = list(range(n))
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c


def numPySum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    return a + b


size = int(sys.argv[1])
# size = 100000

start = time.time()
vectorSum(size)
end = time.time()
print(end - start)

start = time.time()
numPySum(size)
end = time.time()
print(end - start)
