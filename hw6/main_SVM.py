# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:04:24 2020

@author: Alex
"""

# import libraries
import time as time
import pandas as pd
import numpy as np
import os.path
import itertools
import matplotlib.pyplot as plt
from svm import *
from results import *
#from logReg import *

def runSVM_CV(dataCV):
    # Using current time 
    t_st = time.time()
    
    lrs = [10**0, 10**-1, 10**-2, 10**-3, 10**-4]; #intiial learning rates
    Cs = [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2,]; #initial tradeoffs
    hps = list(itertools.product(lrs, Cs))
    best_perf = pd.DataFrame(columns=['Ep','lr', 'C', 'acc', 'obj']); 
    T = 50;
    
    for f in dataCV:
        print('\n Fold -', f)
        data = dataCV[f]
        acc0 = 0; # reset accuracy
        
        for lr, C in hps: # for learning rates and tradeoff combinations
            
            tau = 0.01*C; # early stop threshold
            w_best, best_acc, lc, obj, losses = svm(data, lr, C, tau, T)
            
            if best_acc > acc0:
                best_perf.loc[f] = [len(lc), lr, C, best_acc, obj[-1]]
                acc0 = best_acc
            
    print('\n -- Best Performance over CV Folds -- \n', best_perf)        
        
    t_en = time.time()
    print('\nRuntime (m):', np.round((t_en - t_st)/60,3))
    
    return best_perf

svm_bestHP = runSVM_CV(dataCV);

#%% train with best HP

def runSVM_trn(dataTrn, lr, C, tau, T):
    
    w_best, best_acc, lc, obj, losses = svm(data, lr, C, tau, T)
        
    return w_best, acc0, lc, obj, losses

bestLr = 0.001; bestC = 1000; bestTau = 0.01*bestC; T = 100;
trnW, trnAcc, trnLC, trnObj, trnLosses = runSVM_trn(dataTrn, bestLr, bestC, bestTau, T)
     
plot_learning(trnLC, trnObj, bestLr, bestC, bestTau, 'trnLearning1.pdf')
plot_loss(trnLosses, bestLr, bestC, bestTau, 'trnLoss1.pdf')
#%% test with best weight vector

def runSVM_test(dataTst, w):
    
    data_np = dataTst.to_numpy() # split data
    y = data_np[:,0]
    X = data_np[:,1:]
    X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias
    
    acc = accuracy(X,y,w);
    
    print('Test accuracy: {:.4f}'.format(acc))
    
    return acc

tstAcc = runSVM_test(dataTst, trnW)

#%%