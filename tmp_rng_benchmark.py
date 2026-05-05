import time
import numpy as np

def xoshiro_py(s):
    s0, s1, s2, s3 = s
    res = (s0 + s3) & ((1 << 64) - 1)
    t = (s1 << 17) & ((1 << 64) - 1)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = ((s3 << 45) & ((1 << 64) - 1)) | (s3 >> (64 - 45))
    s[:] = [s0, s1, s2, s3]
    return res


def xoshiro_rand(s):
    return xoshiro_py(s) / 18446744073709551616.0

n = 2000000
s = [1321861022983091513, 3123198108391880477, 1451815097307991481, 5520930533486498032]
total = 0.0
start = time.perf_counter()
for _ in range(n):
    total += xoshiro_rand(s)
end = time.perf_counter()
print('xoshiro_py', end - start)

s = [1321861022983091513, 3123198108391880477, 1451815097307991481, 5520930533486498032]
total = 0.0
start = time.perf_counter()
for _ in range(n):
    total += np.random.random()
end = time.perf_counter()
print('np_random', end - start)
