# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:20:09 2020

@author: Alex
"""

import numpy as np
import math
import pandas as pd

#%% load training data
print('--- Experiments Q2b ---')

'''
Using Adult data set from UCI ML repo (file a1a.train). 14 features: 6 continuous,
8 categorical. Continuous features are discretized. Categorical features with 
m categories are converted to m binary features. 
'''

# Using readlines() 
file1 = open('data/a1a.train', 'r') 
Lines = file1.readlines() 

maxIx = 119;
X = pd.DataFrame(np.zeros((len(Lines),maxIx+1)))
  
r = 0; m = 0;
for line in Lines: 
    X.iloc[r][0] = int(line[:2]) # instance label     
    features = line[3:-2].split() # instance features
    
    for d in features:
        d1 = d.split(':')
        ix = int(d1[0])
        #m = np.maximum(m,int(d1[0]))        
        X.iloc[r][ix+1] = 1        

    r += 1

X = X.rename(columns={0: 'Label'})
y = X.Label
#%%
'''
Implement the decision tree data structure and the ID3 algorithm for
your decision tree.
https://stackoverflow.com/questions/11479624/is-there-a-way-to-guarantee-hierarchical-output-from-networkx
'''

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# read in training data
S = pd.read_excel('data/dt_data.xlsx', sheet_name='data')

y = S['Label']; # assign labels
X = S.drop(columns=['Label']); # drop uneccessary columns
A = list(X.columns)

#%% label probabilities

def labelP(y):
    labels = list(set(list(y.values))); p = [];
    for lbl in labels:
        p.append(list(y.values).count(lbl)/len(y)) # probabilty of Yes label
    return p

p = labelP(y)
# define entropy
import math

def entropy(p):    
    H = 0
    for p_lbl in p:                
        H += -p_lbl * math.log2(p_lbl)
    return(H)

print('Entropy: {:.4f}'.format(entropy(p)))

#%% define information gain

def infoGain(S,A):
    A_prob = [] # attribute probabilities
    A_names = {}; # attribute names   
    y = S['Label']; # get data labels
    H = entropy(labelP(y)) # calculate entropy
    ss_v = [];
    for a in A:        
        V = list(set(S[a].values)) # possible values in attribute
        S_a = S[a] # feature column    
        a0 = []; s0 = [];
        for v in V:
            Sv = S[S_a==v] # return subset values             
            y_av = Sv['Label'] 
            y_av_lbl = list(set(y_av))
            s0.append(len(y_av)); # len of subset
            a1 = [];
            for lbl in y_av_lbl:            
                a1.append(list(y_av.values).count(lbl)/(len(y_av))) # probabilty of Yes label
                A_names[a,v,lbl] = [a1, len(y_av)]     
                            
            a0.append(list(a1))
        
        ss_v.append(s0) # subset probabilities
        A_prob.append(a0) # attribute probabilities                                   
        
    i = 0; A_all = {}; G_all = {};    
    for att in A_prob:        
        I = 0;     
        for k in range(len(att)):                          
            I += (ss_v[i][k]/len(S)) * entropy(att[k]) #id3 update                       
        
        A_all[A[i]] = A_prob[i] # store all A probs
        G_all[A[i]] = (H - I) # information gain      
        i += 1;
            
    return G_all, A_all, A_prob

Gain, A_all, A_prob = infoGain(S,A)
print(Gain)

#%% id3 build tree

import networkx as nx
from collections import Counter

# initialize tree structure
G = nx.DiGraph(); tree = {};
def id3(S,A,b):
    
    # if all examples have the same label return a single node tree
    lbls = list(set(S.Label))
    if len(lbls) == 1: 
        G.add_node(lbls[0]) 
        tree['single'] = lbls[0]
            
    # otherwise we need to build an actual tree
    else:
        # return information gain among attributes in A and add 'Label' for InfoGain function        
        Gain, _, _ = infoGain(S,A) 
        best = max(Gain,key=Gain.get) # identify best attribute

        # add subtree
        if b != 'none':
            tree[b[0]][b[1]] = {best}
            G.add_edge(b[0],best)
            print('Add \|/', b[0],'[', b[1],']')
        
        # add root node
        G.add_node(best)        
        print('Root:', best)
        tree[best] = {}
        
        # for each possible value v in attribute
        v_in_A = list(set(S[best]))
        for v in v_in_A:          
            # create subset of examples in S with A = v
            Sv = S[S[best] == v]
            Sv_lbl = list(set(Sv.Label))
            print('  Add /', best ,'[', v, ']')
            tree[best][v] = {}
            
            # if empty subset, then add most common label
            if len(Sv_lbl) == 0:  
                maj_lbl, _ = Counter(Sv['Label']).most_common()[0] # return majority label in Sv
                maj_lbl0 = str(v)+' = '+maj_lbl
                print('  Add <>', maj_lbl,'- for empty subset of [', best,']')
                G.add_node(maj_lbl0)
                G.add_edge(best,maj_lbl0)
                tree[best][v] = maj_lbl                
            
            # add leaf node
            elif len(Sv_lbl) == 1:    
                leaf = Sv['Label'].values[0]
                leaf0 =  str(v)+' = '+leaf
                print('  Add <>', best , '[', v, '] =', leaf)                
                G.add_node(leaf0)
                G.add_edge(best,leaf0)        
                tree[best][v] = leaf                
            
            # else build subtree with S - previous best attribute
            else:
                # remove used attribute from list and update     
                Anew = A.copy()
                Anew.remove(best)                                
                
                # branch connecting current root to next root                 
                b = [best,v]    
                # recurse subtree
                print('--->')
                id3(Sv, Anew, b)
    
    return G, tree

A = list(S.columns)[0:-1]
G, tree = id3(S,A,'none')
print('\n',tree)

pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True, arrows=True, node_size=1200, node_color='white',
        font_size=14, font_family='Times New Roman', font_color='k')


#%% ID3 ref



#%% define ID3 information gain

def infoGain(S):
    A = [] # attribute probabilities
    A_names = {}; # attribute names
    Sv = [] # subset sizes
    X = S.drop(columns=['Label']); # training attributes
    y = S['Label']; # training labels
    for a in X.columns:
        V = list(set(X[a].values)) # possible values in attribute
        X_a = X[a] # feature column    
        a0 = []; s0 = []
        for v in V:
            X_av = X_a[X_a==v]            
            y_av = y.iloc[list(X_av.index)]      
            y_av_lbl = list(set(y_av))
            s0.append(len(y_av)); # Sv = len of subset
            a1 = [];
            for lbl in y_av_lbl:            
                a1.append(list(y_av.values).count(lbl)/(len(y_av))) # probabilty of Yes label
                A_names[a,v,lbl] = [a1, len(y_av)]
            
            a0.append(list(a1))
        
        Sv.append(s0)
        A.append(a0)
    
    # calculate information gain by entropy
    k = 0; Gain = {}
    H = entropy(labelP(y))
    for att in A:
        I = 0;     
        for i in range(len(att)):            
            I += (Sv[k][i]/len(S)) * entropy(att[i]) #id3 update   
        Gain[Attributes[k]] = (H - I) # information gain
        
        #print(Attributes[k],': Info Gain = {:.4f}'.format((H - I)))
        k += 1;
    
    A_all = {}
    for i in range(len(A)):
        A_all[Attributes[i]] = A[i]
    
    return Gain, A_all, Sv

Attributes = A;
Gain, A_all, Sv0 = infoGain(S)
print(Gain)


#%% draw graph    
#https://stackoverflow.com/questions/11479624/is-there-a-way-to-guarantee-hierarchical-output-from-networkx
import networkx as nx
import matplotlib as plt

plt.style.use('ggplot')

tree={}; tree[0]={1,2}; tree[1]={3,4,5}
G = nx.DiGraph(tree)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, arrows=True, node_size=500,
        font_size=12, font_family='Times New Roman', font_color='w')

#%% cross-validation
'''
The depth of a tree is a hyper-parameter to the DT algorithm that reduces
overfitting. Depth = max path length from root to any leaf. Our goal is to 
discover good hyperparameters using the training data only. To determine if 
the learned hyperparameter is good or not, we can set aside some of the training
data into a validation set. When training is complete, we test the results on
the validation data. 

However, since we did not train on the whole dataset, we may have introduced 
a statistical bias in the classier caused by the choice of the validation set. 
To correct for this, we will need to repeat this process multiple times for 
different choices of the validation set.

For this problem we will implement k-folds cross-validation.

'''





























