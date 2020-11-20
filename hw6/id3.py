# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:27:32 2020

@author: Alex
"""

import numpy as np
np.random.seed(42)

def prob(y):
    labels = y.unique(); p = [];
    for lbl in labels:
        p.append(list(y.values).count(lbl)/len(y)) # probabilty of Yes label
    return p

def entropy_Lbl(p):
    H = 0
    for pi in p:        
        H += -pi * np.log2(pi)
    return H

def entropy_A(df):

    lblsIn = df.Label.unique()    
    #entA = np.zeros(len(df.columns))
    entA = {}
    
    for attribute in df.columns[1:]:
        valsInA = df[attribute].unique()
        H_A = 0;
        for v in valsInA:
            H_Av = 0;
            
            for lbl in lblsIn:
                
                sizeSv = len(df[attribute][df[attribute]==v][df.Label == lbl]) 
                sizeS = len(df[attribute][df[attribute]==v])
                
                if sizeSv > 0:
                    p_Av = sizeSv/sizeS;            
                    H_Av += -p_Av * np.log2(p_Av)
                                    
            fracS = sizeS/len(df)
            
            H_A += fracS*H_Av       
        

        entA[attribute] = H_A
        
    return entA

def infoGain(entLbl, entA):
    igA = {}
    # calculate information gain
    for k,v in entA.items():
        igA[k] = entLbl - v
        
    # return features list sorted by information gain
    return sorted(igA.items(), key=lambda x: x[1] , reverse=True)

#df = dataTrn.sample(int(0.004*len(dataTrn)), replace=True)

entLbl = entropy_Lbl(prob(df1.Label))
entA = entropy_A(df1)

igA = infoGain(entLbl, entA)

#%%



def id3(df, df0, attributes, depth, parent=None):
    # if all examples have the same label return a single node tree    
    if len(np.unique(df.Label)) <= 1: 
        return np.unique(df.Label)[0]
    
    # if subset is empty, return most likely label
    elif len(df) == 0:
        values, counts = np.unique(df0.Label, return_counts=True)
        # if equal number of labels, assume a label of -1
        return [counts.argmax() if counts.argmax() != 0 else -1][0]
    
    # if no attributes left, return parent
    elif len(attributes) == 0:
        
        return parent
    
    # grow subtree
    else:
        # set the default feature label
        values, counts = np.unique(df.Label, return_counts=True)
        parent = [counts.argmax() if counts.argmax() != 0 else -1][0]
        
        entLbl = entropy_Lbl(prob(df.Label))
        lblAtts = ['Label'] + attributes
        entA = entropy_A(df[lblAtts])
        attGain = infoGain(entLbl, entA)
        bestAtt = attGain[0][0]
                
        # add node for best attribute
        print('    Add node->',bestAtt)
        tree = {bestAtt:{}}
        depth += 1;
        if depth >= maxDepth:           
            return tree
        
        # remove bestAtt from df
        attributes.remove(bestAtt)
        
        for v in np.unique(df[bestAtt]):
            Sv = df.loc[df[bestAtt] == v]
            
            # build subtree on Sv, with remaining features from df.colums (excluding label)
            subtree = id3(Sv, df, attributes, depth, parent)
            
            tree[bestAtt][v] = subtree
            
    return tree

def endLeaf(tree):
    print('    </> add leaves')
    for k,v in tree.items():                             
        # if empty
        if not bool(v):
            for attVal in [0.0,1.0]:
                values, counts = np.unique(df.loc[df[k] == int(attVal)].Label, return_counts=True)
                # if equal number of labels, assume a label of -1
                tree[k][attVal] = [counts.argmax() if counts.argmax() != 0 else -1][0]
                        
        else:             
            if isinstance(tree[k],dict):
                endLeaf(tree[k])
            else: pass
        
    return tree


trees = {}      
depths = [1,2,4,8]
t_st = time.time()

df = dataTrn.sample(int(0.1*len(dataTrn)), replace=True)
df = df.reset_index(drop=True)

for maxDepth in depths:
    trees[maxDepth] = {}
    print('\nMax Depth:', maxDepth)
    
    for i in np.arange(200):
        print('>> Tree:', i)
        attributes = list(df.columns[1:])
        # input id3(df, df0, attributes, depth, maxDepth, parent=None):
        prunedTree = id3(df,df,attributes,0,maxDepth)
        trees[maxDepth][i] = endLeaf(prunedTree) #add end leaf labels to tree
        
        print(trees[maxDepth][i])
        
t_en = time.time()
print('\nRuntime (m):', np.round((t_en - t_st)/60,3))

#%%
df = dataTrn.sample(int(0.1*len(dataTrn)), replace=True)
df = df.reset_index(drop=True)
#df1 = df[['Label',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]] 

#%%


#tree0 = trees[1][0]            
#tree4 = endLeaf(trees[2][0])