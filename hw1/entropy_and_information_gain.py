# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:12:46 2020

@author: Alex
"""

import numpy as np
import math

#%% calculate entropy
print('--- Decision Trees Q2b ---')

p = [5/9, 4/9]
p = [1/2, 1/2]

def entropy(p):
    H = 0
    for pi in p:        
        H += -pi * math.log2(pi)
    return H

H = entropy(p)
print('Entropy: {:.4f}'.format(H))

#%% calculate information gain by ID3

print('\n --- Decision Trees Q2c, Q3a, and Q3b ---')

features = ['Friday', 'Hungry','Patrons', 'Type']
for f in features:        
    S = 9;
    print('\nFeature: ', f)
    if f == 'Friday':
        A = [[1/3,2/3],[4/6,2/6]]
        Sv = [3,6]

    if f == 'Hungry':
        A = [[4/5,1/5],[1/4,3/4]]
        Sv = [5,4]

    if f == 'Patrons':
        A = [[1/4,3/4],[4/4],[1/1]]
        Sv = [4,4,1]

    if f == 'Type':
        A = [[1/3,2/3],[1/2,1/2],[1/1],[2/3,1/3]]
        Sv = [3,2,1,3]

    # calculate information gain by misclassification rate
    for ai in A:
        I = 0;
        for i in range(len(ai)):
            I += (Sv[i]/S) * entropy(ai)
    G = (H - I)
        
    print('Information Gain by ID3: {:.4f}'.format(G))

    # calculate information gain by misclassification rate
    M = 1 - np.max(p)
    #Ms = 1 - np.max(A)[0]    
    Gm = M - (np.max(Sv)/9)*M
    print('Information Gain by Missclassification: {:.4f}'.format(Gm))

    #% calculate information gain by GINI
    GINI = 0
    for pi in p:        
        GINI += pi * (1 - pi)
    
    for ai in A:
        Gs = 0;
        for i in range(len(ai)):
            Gs += (Sv[i]/S) * (ai[i]*(1 - ai[i]))
    Gg = (GINI - Gs)

    print('Information Gain by GINI: {:.4f}'.format(Gg))

#%% calculate information gain by ID3

print('\n --- Decision Trees Q2c, Q3a, and Q3b ---')

features = ['x1', 'x2','x3', 'x4']
for f in features:        
    S = 4;
    print('\nFeature: ', f)
    if f == 'x1':
        A = [[2/3,1/3],[1/1]]
        Sv = [3,1]

    if f == 'x2':
        A = [[2/2],[2/2]]
        Sv = [2,2]

    if f == 'x3':
        A = [[1/2,1/2],[1/2,1/2]]
        Sv = [2,2]

    if f == 'x4':
        A = [[1/3,2/3],[1/1]]
        Sv = [3,1]

    # calculate information gain by misclassification rate
    for ai in A:
        I = 0;
        print(f, ai)
        for i in range(len(ai)):
            I += (Sv[i]/S) * entropy(ai)
    G = (H - I)
        
    print('Information Gain by ID3: {:.4f}'.format(G))


#%%
3/4*(-2/3*np.log2(2/3)-1/3*np.log2(1/3))



