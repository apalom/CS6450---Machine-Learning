# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:27:32 2020

@author: Alex
"""

import numpy as np
np.random.seed(42)

# calculate label probability
def prob(y):
    labels = y.unique(); p = [];
    for lbl in labels:
        p.append(list(y.values).count(lbl)/len(y)) # probabilty of Yes label
    return p

# calculate label entropy
def entropy_Lbl(p):
    H = 0
    for pi in p:        
        H += -pi * np.log2(pi)
    return H

# calculate attribute entropy
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
                
                #sizeSv = len(df[attribute][df[attribute]==v][df.Label == lbl]) 
                #sizeSv = len(df[df[attribute] == v][df.Label == lbl]) 
                #sizeS = len(df[attribute][df[attribute]==v])
                #temp = df[df[attribute] == v]
                #sizeSv = len(temp[temp.Label == lbl])
                S = df[df[attribute] == v]
                sizeSv = len(S[S.Label == lbl])                 
                sizeS = len(S)
                
                if sizeSv and sizeS > 0:
                    p_Av = sizeSv/sizeS;            
                    H_Av += -p_Av * np.log2(p_Av)
                                    
            fracS = sizeS/len(df)
            
            H_A += fracS*H_Av       
        
        entA[attribute] = H_A
        
    return entA

# calculate information gain
def infoGain(entLbl, entA):
    igA = {}
    # calculate information gain
    for k,v in entA.items():
        igA[k] = entLbl - v
        
    # return features list sorted by information gain
    return sorted(igA.items(), key=lambda x: x[1] , reverse=True)

#df = dataTrn.sample(int(0.004*len(dataTrn)), replace=True)

#entLbl = entropy_Lbl(prob(df1.Label))
#entA = entropy_A(df1)
#igA = infoGain(entLbl, entA)

#%% run id3 algorithms

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
        #print('    Add node->',bestAtt)
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

# add end leafs (labels) to terminated branches based on most common label
def endLeaf(tree):
    #print('    </> add leaves')
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

#%% dt prediction

# return label prediction for each data sample and decision tree
def predDT(sample, tree):
    
    # search over all features in data sample xi
    for feature in range(len(sample)):
        
        # if the feature was used in the DT, traverse tree
        if feature in list(tree.keys()):
            
            result = tree[feature][sample[feature]]
            
            # if result is a dictionary... traverse that subtree
            if isinstance(result,dict):
                return predDT(sample,result)

            else:
                return result
       
def transformData(data, trees, depths):
    # now transform data based on 200 trees
    data_np = data.to_numpy() # convert to NumPy for computational efficiency
    y = data_np[:,0]
    X = data_np[:,1:]
    
    dataTrfm = {}
    
    # over all tree depths
    for maxDepth in depths:
        print('\n++ Tree Depth: ', maxDepth)
        treesDeep = trees[maxDepth]
        dataTrfm[maxDepth] = np.zeros((X.shape[0],200)) # add additional column for bias
        
        # over all data samples
        for i in range(X.shape[0]):
            if np.mod(i,100) == 0:
                print('.', end=" ") 
                    
            xi = X[i]       
            
            # over all trees
            for t in range(0,200):                
                dt = treesDeep[t]            
                dataTrfm[maxDepth][i,t] = predDT(xi, dt)   
                
        # bias weight will be added at SVM algorithm
        dataTrfm[maxDepth] = np.insert(dataTrfm[maxDepth], 0, y, axis=1)
        
    return dataTrfm

# print('\n\nEnsemble Training Data')
# dataTrfm_trn = transformData(dataTrn, trees, depths)
# print('\n\nEnsemble Testing Data')
# dataTrfm_tst = transformData(dataTst, trees, depths)
# print('\nEnsemble Cross-Validation Data')
# dataTrfm_CV = {}
# for fold in dataCV:
#     print('\n   Fold:', fold)
#     dataTrfm_CV[fold] = transformData(dataCV[fold], trees, depths)

#%% output transformed data

for maxDepth in depths:
    path = 'data/dt_transformed/depth'+str(maxDepth)+'.csv'
    np.savetxt(path,dataTrfm[maxDepth],delimiter=",")


