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
#featues = ['Type']
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

    # calculate information gain by ID3  
    I = 0; i = 0;
    for ai in A:        
        I += (Sv[i]/S) * entropy(ai)
        i+=1;

    G = (H - I)
        
    print('Information Gain by ID3: {:.4f}'.format(G))

    # calculate information gain by misclassification rate   
    M = 1 - np.max(p)
    Ms = 0; i = 0;
    for ai in A:
        Ms += (Sv[i]/S) * (1-np.max(ai))
        i+=1;
    
    Gm = M - Ms
    
    print('Information Gain by Miss: {:.4f}'.format(Gm))

    #% calculate information gain by GINI
    GINI = 0
    for pi in p:        
        GINI += pi * (1 - pi)
    
    Gs = 0; k = 0;
    for ai in A:        
        for i in range(len(ai)):
            Gs += (Sv[k]/S) * (ai[i]*(1 - ai[i]))
        k += 1;
        
    Gg = (GINI - Gs)

    print('Information Gain by GINI: {:.4f}'.format(Gg))






