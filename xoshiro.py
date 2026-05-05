import numpy as np
from numba import njit


@njit
def rotl(a, w):
    return a << w | a >> (64 - w)


@njit
def xoshiro(s):
    """Uses Xoshiro to generate PRNs"""
    res = np.uint64(s[0] + s[3])

    t = np.uint64(s[1] << 17)
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = rotl(s[3], 45)

    return res


s = np.zeros(4, dtype=np.uint64)
s[0] = 1321861022983091513
s[1] = 3123198108391880477
s[2] = 1451815097307991481
s[3] = 5520930533486498032

