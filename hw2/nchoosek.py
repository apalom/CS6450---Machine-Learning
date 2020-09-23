# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:00:49 2020

@author: Alex
"""

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  

n, K = (4,2); size = 0;
for k in range(1,K+1):
    print(k)
    size += ncr(n,k)

print("Size:", size)