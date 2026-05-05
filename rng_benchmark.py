import time
import numpy as np
from xoshiro import rand_xoshiro, s



n = 2000000
total = 0.0
start = time.perf_counter()
for _ in range(n):
    total += rand_xoshiro(s)
end = time.perf_counter()
print('xoshiro_py', end - start)
print(total)

total = 0.0
start = time.perf_counter()
for _ in range(n):
    total += np.random.random()
end = time.perf_counter()
print('np_random', end - start)
print(total)
