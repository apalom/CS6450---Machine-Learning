# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:56:50 2020

@author: Alex
"""

import numpy as np

# check if data is pure or trivial
def checkSame(data):
        
    lblsIn = np.unique(data[:,0]) # column 0 is label column

    if len(lblsIn) == 1:
        return True
    else: return False

# return most common label in data    
def likeliestLabel(data):
        
    # count labels in column 0 (label column)
    lblsIn, countLbls = np.unique(data[:,0], return_counts=True)

    index = countLbls.argmax()
    label = lblsIn[index]
    
    return label

# return subset of data with feature value == value
def split(data, feature, value):
    
    featureVals = data[:, feature]

    dataLeft = data[featureVals == value]
    dataRight = data[featureVals !=  value]
    
    return dataLeft, dataRight

# calculate label entropy
def entropyLbl(data):
    
    # column 0 is label column
    _, counts = np.unique(data[:,0], return_counts=True)

    p = counts / counts.sum() # get label probabilities
    entropy = sum(p * -np.log2(p))
     
    return entropy

# calculate feature entropy
def entropyS(dataLeft, dataRight):
    
    n = len(dataLeft) + len(dataRight)
    p_dataLeft = len(dataLeft) / n
    p_dataRight = len(dataRight) / n

    subsetEntropy =  (p_dataLeft * entropyLbl(dataLeft) 
                      + p_dataRight * entropyLbl(dataRight))
    
    return subsetEntropy

# best root determined by greatest information gain (lowest entropy)
def bestRoot(data):

    # find lowest entropy by updating this value with improvements over all features
    bestEntropy = 1000;
    features = np.arange(1,data.shape[1]) # index of non-label columns
    for f in features: # over all features
        for v in [0,1]: # over possible values
            dataIn, dataOut = split(data, f, v)
            featureEntropy = entropyS(dataIn, dataOut)
            
            # update entropy
            if featureEntropy < bestEntropy:
                bestEntropy = featureEntropy;
                bestFeature = f;
                bestValue = v;
                
    return bestFeature, bestValue, bestEntropy

def id3(data, maxDepth, depth):         
    
    # base case
    # if all examples have the same label return a single node tree  
    # if dt depth is greater than max depth      
    if checkSame(data) or depth >= maxDepth:
        label = likeliestLabel(data)       
        return label
    
    # grow subtree
    else:    
        
        depth += 1;
        
        # calculate best root for splitting
        splitFeature, splitVal, _ = bestRoot(data)     
        # dataLeft is subset with values = splitVal
        dataLeft, dataRight = split(data, splitFeature, splitVal)
        
        # create sub-tree
        root = splitFeature       
        tree = {root:{}}
                
        # recurse down left and right branch (data subsets)        
        branchLeft = id3(dataLeft, maxDepth, depth)  
        branchRight = id3(dataRight, maxDepth, depth)
        
        tree[root][splitVal] = branchLeft
        tree[root][1-splitVal] = branchRight       
        
        return tree

#%%

def predictLbl(sample, tree):
    # root feature for splitting   
    feature = list(tree.keys())[0];
    # shift correction after removal of label
    value = sample[feature-1];
        
    result = tree[feature][value]
    
    # if result is a dictionary... traverse that subtree
    if isinstance(result,dict):
        return predictLbl(sample,result)

    else: # else return the label
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
        dataTrfm[maxDepth] = np.zeros((X.shape[0],len(trees[1]))) # initialize
        
        # over all data samples
        for i in range(X.shape[0]):
            if np.mod(i,100) == 0:
                print('.', end=" ") 
                    
            xi = X[i]       
            
            # over all trees
            for t in range(0,len(trees[1])):                
                dt = treesDeep[t]            
                dataTrfm[maxDepth][i,t] = predictLbl(xi, dt)   
                                    
        # bias weight will be added at SVM algorithm
        # insert y labels as column 1 of dataTrfm
        dataTrfm[maxDepth] = np.insert(dataTrfm[maxDepth], 0, y, axis=1)
        
    return dataTrfm
