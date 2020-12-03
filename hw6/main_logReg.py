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
from results import *
from logReg import *

def runLogReg_CV(dataCV, es):
    # Using current time 
    t_st = time.time()
    
    lrs = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]; #intiial learning rates
    sig2s = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4,]; #initial tradeoffs
    hps = list(itertools.product(lrs, sig2s))
    best_perf = pd.DataFrame(columns=['Ep','lr', 'sig2', 'acc', 'obj']); 
    T = 10;
    
    for f in dataCV:
        print('\n Fold -', f)
        # data prep for validation fold
        dataVal = dataCV[f]['val'].to_numpy()
        X_val = dataVal[:,1:]; X_val = np.hstack((X_val, np.ones((X_val.shape[0],1)))); # add bias
        y_val = dataVal[:,0];            
        # data fold for training
        data = dataCV[f]['trn'];      
        acc0 = 0; # reset accuracy
        
        for lr, sig2 in hps: # for learning rates and tradeoff combinations
            
            # No early stop threshold for CV: es = 'None'
            # CV training
            w_best, _, lc, obj, losses = logReg(data, lr, sig2, es, T)
            # CV validation 
            acc_Val = accuracy(X_val, y_val, w_best) # accuracy(X,y,w):
        
            if acc_Val > acc0:
                best_perf.loc[f] = [len(lc), lr, sig2, acc_Val, obj[-1]]
                acc0 = acc_Val
            
    print('\n -- Best Performance over CV Folds -- ')
    print(best_perf)        
    print('\nEarly stop:', es)      
    t_en = time.time()
    t_run = np.round((t_en - t_st)/60,3)
    print('\nRuntime (m):', t_run)
    
    return best_perf, t_run

reps = {}; repeats = 3; runtimes = {}; 
es = 'None'; avgObj = 0
for r in range(repeats):
    # input dataCV and early stopping factor
    logReg_bestHP, t_run = runLogReg_CV(dataCV, es);
    avgObj += logReg_bestHP.obj.mean();
    reps[r] = logReg_bestHP;
    runtimes[r] = t_run    

# average cross validation objective value for early stopping definition
avgObj = int(avgObj/repeats)

#%% train with best HP

def runLogReg_trn(dataTrn, lr, sig2, tau, T):
    
    w_best, best_acc, lc, obj, losses = logReg(dataTrn, lr, sig2, tau, T)
        
    return w_best, best_acc, lc, obj, losses

bestLr = 0.1; bestSig2 = 10000; T = 100;
# early stop condition = 1% of avg. CV objective
bestTau = 0.01*730#avgObj;

logReg_Trn = {}
logReg_Trn['w'], logReg_Trn['Acc'], logReg_Trn['LC'], logReg_Trn['Obj'], logReg_Trn['Losses'] = runLogReg_trn(dataTrn, bestLr, bestSig2, bestTau, T)
print('\n    Accuracy: {:.3f}'.format(logReg_Trn['Acc']))
 
plot_learning(logReg_Trn['LC'], logReg_Trn['Obj'], bestLr, bestC, bestTau, 'logReg_trnLearning.pdf')
plot_loss(logReg_Trn['Losses'], bestLr, bestC, bestTau, 'logReg_trnLoss.pdf')

#% test with best weight vector

def runLogReg_tst(dataTst, w):
    
    data_np = dataTst.to_numpy() # split data
    y = data_np[:,0]
    X = data_np[:,1:]
    X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias
    
    acc = accuracy(X,y,w);
    
    print('\nTest accuracy: {:.3f}'.format(acc))
    
    return acc

tstAcc = runLogReg_tst(dataTst, logReg_Trn['w'])
