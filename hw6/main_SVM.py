# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:04:24 2020

@author: Alex
"""

# import libraries
import time as time
import pandas as pd
import numpy as np
np.random.seed(42)
import os.path
import itertools
import matplotlib.pyplot as plt
from svm import *
from results import *

def runSVM_CV(dataCV, es):
    # Using current time 
    t_st = time.time()
    
    lrs = [10**0, 10**-1, 10**-2, 10**-3, 10**-4]; #intiial learning rates
    Cs = [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2,]; #initial tradeoffs
    hps = list(itertools.product(lrs, Cs))
    best_perf = pd.DataFrame(columns=['Ep','lr', 'C', 'acc', 'obj']); 
    T = 10;
    
    for f in dataCV:
        print('\n Fold -', f)
        dataVal = dataCV[f]['val'].to_numpy()
        data = dataCV[f]['trn'];         
        acc0 = 0; # reset accuracy
        
        for lr, C in hps: # for learning rates and tradeoff combinations
            
            # CV training
            w_best, _, lc, obj, losses = svm(data, lr, C, es, T)
            # CV validation
            X = dataVal[:,1:]; X = np.hstack((X, np.ones((X.shape[0],1)))); # add bias
            y = dataVal[:,0];             
            acc_Val = accuracy(X,y,w_best) # accuracy(X,y,w):
        
            if acc_Val > acc0:
                best_perf.loc[f] = [len(lc), lr, C, acc_Val, obj[-1]]
                acc0 = acc_Val
            
    print('\n -- Best Performance over CV Folds -- ')
    print(best_perf)     
    print('\nEarly stop:', es)          
    t_en = time.time()
    t_run = np.round((t_en - t_st)/60,3)
    print('\nRuntime (m):', t_run)
    
    return best_perf, t_run

# repeat cross validation
reps = {}; repeats = 3; runtimes = {}; 
es = 'None'; avgObj = 0
for r in range(repeats):
    # input dataCV and early stopping factor
    svm_bestHP, t_run = runSVM_CV(dataCV, es);
    avgObj += svm_bestHP.obj.mean();
    reps[r] = svm_bestHP;
    runtimes[r] = t_run    

# average cross validation objective value for early stopping definition
avgObj = int(avgObj/repeats)

#%% train with best HP

def runSVM_trn(dataTrn, lr, C, tau, T):
    
    w_best, best_acc, lc, obj, losses = svm(dataTrn, lr, C, tau, T)
        
    return w_best, best_acc, lc, obj, losses

bestLr = 0.0001; bestC = 1000; bestTau = int(0.01*avgObj); T = 100;
svm_Trn = {}
svm_Trn['w'], svm_Trn['Acc'], svm_Trn['LC'], svm_Trn['Obj'], svm_Trn['Losses'] = runSVM_trn(dataTrn, bestLr, bestC, bestTau, T)
     
plot_learning(svm_Trn['LC'], svm_Trn['Obj'], bestLr, bestC, bestTau, 'svm_trnLearning.pdf')
plot_loss(svm_Trn['Losses'], bestLr, bestC, bestTau, 'svm_trnLoss.pdf')

#%% test with best weight vector

def runSVM_test(dataTst, w):
    
    data_np = dataTst.to_numpy() # split data
    y = data_np[:,0]
    X = data_np[:,1:]
    X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias
    
    acc = accuracy(X,y,w);
    
    print('Test accuracy: {:.4f}'.format(acc))
    
    return acc

tstAcc = runSVM_test(dataTst, svm_Trn['w'])

#%%