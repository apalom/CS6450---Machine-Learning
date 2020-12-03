# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:04:24 2020

@author: Alex

Ensemble Learning: SVM on id3 
"""

# import libraries
import time as time
import pandas as pd
import numpy as np
import os
import os.path
import itertools
import matplotlib.pyplot as plt
from id3 import *
from id3_improved import *
from results import *

#%% use training set to build decision trees
trees = {}      
depths = [1,2];#[1,2,4,8]
t_st = time.time()

for maxDepth in depths:
    trees[maxDepth] = {}
    print('\nMax Depth:', maxDepth)
    
    for i in np.arange(200):
        df = dataTrn.sample(int(0.1*len(dataTrn)), replace=True)
        
        #print('>> Tree:', i)
        attributes = list(df.columns[1:])
        # input id3(df, df0, attributes, depth, maxDepth, parent=None):
        prunedTree = id3(df,df,attributes,0,maxDepth)
        trees[maxDepth][i] = endLeaf(prunedTree,df) #add end leaf labels to tree
        
        #print(trees[maxDepth][i])
        
        if np.mod(i,10) == 0:
            print('.', end=" ") 
        
t_en = time.time()
print('\nRuntime (m):', np.round((t_en - t_st)/60,3))
   

#%% transformer data

depths = [1,2]
print('\n\nEnsemble Training Data')
dataTrfm_trn = transformData(dataTrn, trees, depths)
print('\n\nEnsemble Testing Data')
dataTrfm_tst = transformData(dataTst, trees, depths)

print('\nEnsemble Cross-Validation Data')
dataTrfm_CV = {}
for f in dataCV:
    print('\n\n   Fold:', f)
    dataTrfm_CV[f] = {}
    print('-Training')
    dataTrfm_CV[f]['trn'] = transformData(dataCV[f]['trn'], trees, depths)
    print('\n-Validation')
    dataTrfm_CV[f]['val'] = transformData(dataCV[f]['val'], trees, depths)

#%%
depths = [1,2]
dataTrfm0 = transformData(dataCV[1]['val'], trees, depths)

# for f in dataTrfm_CV:
#     dataTrfm_CVfold = dataTrfm_CV[f]
#     for d in depths:
#         dataTrfm_CVfoldD = dataTrfm_CVfold['trn'][d]
#             X = dataTrfm_CVfoldD[:,1:]
#             y = dataTrfm_CVfoldD[:,0]
        

#%% run SVM over trees ensemble

def runSVMid3_CV(dataCV, depths, es):
    # Using current time 
    t_st = time.time()
    
    lrs = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]; #intiial learning rates
    Cs = [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2,]; #initial tradeoffs
    #lrs = [0.0001]; Cs = [1000]
    
    hps = list(itertools.product(lrs, Cs))
    best_perf = pd.DataFrame(columns=['Ep','d','lr', 'C', 'acc', 'obj']); 
    T = 10;
        
    for f in [1]:#dataCV:
        print('\n \nFold -', f)
        
        for d in depths:    
            dataVal0 = pd.DataFrame(dataCV[f]['val'][d]) # validation data folds and depths
            dataVal = dataVal0.fillna(-1)        
            dataVal = dataVal.to_numpy()
            
            data0 = pd.DataFrame(dataCV[f]['trn'][d]) # training data folds and depths
            data = data0.fillna(-1)
            acc0 = 0; # reset accuracy
            
            for lr, C in hps: # for learning rates and tradeoff combinations            
                
                # CV training
                w_best, _, lc, obj, losses = svm(data, lr, C, es, T)
                # CV validation
                X = dataVal[:,1:]; X = np.hstack((X, np.ones((X.shape[0],1)))); # add bias here b/c Val doesn't go to SVM
                y = dataVal[:,0];             
                acc_Val = accuracy(X,y,w_best) # accuracy(X,y,w):               
                
                if acc_Val > acc0: # update best performance
                    best_perf.loc[f] = [len(lc), d, lr, C, acc_Val, obj[-1]]
                    acc0 = acc_Val
            
    print('\n -- Best Performance over CV Folds -- ')
    print(best_perf)        
    print('\nEarly stop:', es)      
    t_en = time.time()
    t_run = np.round((t_en - t_st)/60,3)
    print('\nRuntime (m):', t_run)
    
    return best_perf, t_run

reps = {}; repeats = 1; runtimes = {}; 
es = 'None'; avgObj = 0;
for r in range(repeats):
    # input dataCV and early stopping factor
    id3SVM_bestHP, t_run = runSVMid3_CV(dataTrfm_CV, depths, es);
    avgObj += id3SVM_bestHP.obj.mean();
    reps[r] = id3SVM_bestHP;
    runtimes[r] = t_run    

# average cross validation objective value for early stopping definition
avgObj = int(avgObj/repeats)

#%%

for f in dataTrfm_CV:
    for d in depths:
        dataCV[f]['val'][d] = dataCV[f]['val'][d] # validation data folds and depths
        data = pd.DataFrame(dataCV[f]['trn'][d]) # training data folds and depths


#%%
lr = 1; C = 1; T = 25;
data = pd.DataFrame(dataTrfm_CV[1]['trn'][2]) 
w_best, acc, lc, obj, losses = svm(data1, lr, C, es, T)

#%% train with best HP

def runSVMid3_trn(dataTrn, lr, C, tau, T):
    
    dataTrn = pd.DataFrame(dataTrn)
    w_best, best_acc, lc, obj, losses = svm(dataTrn, lr, C, tau, T)
        
    return w_best, best_acc, lc, obj, losses

bestLr = 0.0001; bestC = 1000; bestDepth = 8; bestTau = int(0.01*avgObj); T = 100;
svmID3_Trn = {}
svmID3_Trn['w'], svmID3_Trn['Acc'], svmID3_Trn['LC'], svmID3_Trn['Obj'], svmID3_Trn['Losses'] = runSVMid3_trn(dataTrfm_trn[bestDepth], bestLr, bestC, bestTau, T)
     
plot_learning(svmID3_Trn['LC'], svmID3_Trn['Obj'], bestLr, bestC, bestTau, 'svmID3_trnLearning.pdf')
plot_loss(svmID3_Trn['Losses'], bestLr, bestC, bestTau, 'svmID3_trnLoss.pdf')


#%% test with best weight vector

def runSVMid3_test(dataTst, w):
    
    y = data_np[:,0]
    X = data_np[:,1:]
    X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias
    
    acc = accuracy(X,y,w);
    
    print('Test accuracy: {:.4f}'.format(acc))
    
    return acc

tstAcc = runSVMid3_test(dataTrfm_tst[bestDepth], svmID3_Trn['w'])

#%%