# (c) MJMJ/2020

import numpy as np
from itertools import permutations

def mj_dist_perms(p1, p2):
    d = 0
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            d += 1

    return d

ncells = 4

x = list(range(ncells))

perms = list(permutations(x, ncells))

nperms = len(perms)

# Compute distances
D = np.zeros((nperms, nperms)) -1

for i in range(0, nperms-1):
    for j in range(i+1, nperms):
        D[i, j] = mj_dist_perms(perms[i], perms[j])

# Select the top K
l_sel_perms = [perms[0]]
K = 18

cdist = ncells

while len(l_sel_perms) < K:
    # Could be done more efficiently
    for i in range(0, nperms - 1):
        for j in range(i + 1, nperms):
            if D[i, j] == cdist:
                l_sel_perms.append(perms[j])
                if len(l_sel_perms) >= K:
                    break
        if len(l_sel_perms) >= K:
            break
    cdist -= 1

    if cdist < 2:
        break


print("Done!")