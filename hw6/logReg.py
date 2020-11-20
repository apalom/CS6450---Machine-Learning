# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:56:58 2020

@author: Alex
"""

# import libraries
import pandas as pd
import numpy as np

#%% Logistic Regression model

t_st = time.time()

lrs = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]; #intiial learning rates
sig2s = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4,]; #initial tradeoffs
hps = list(itertools.product(lrs, Cs))
best_perf = pd.DataFrame(columns=['Ep','lr', 'sig2', 'acc', 'obj']); 
T = 25; 

Lr = 1; sig2 = 1; tau = 0.01*sig2

w_LogR, acc_LogR, lc_LogR, obj_LogR, losses_LogR = logReg(dataCV[1], Lr, sig2, tau, T)

#%% logReg model

def logReg(data, g0, sig2s, tau, T):

    print('Logistic Regression Model')    
    print('[-- HP: lr {} + sig2 {} --]'.format(g0,sig2s))
    data_np = data.to_numpy() # split data
    y = data_np[:,0]
    X = data_np[:,1:]
    X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias
    
    w = np.zeros(X.shape[1]); # initialize weights with bias
    acc0 = 0; # initialize accuracy baseline    
    up = 0; # initialize update counters
    idx = np.arange(X.shape[0]); # index for stepping through data
    
    lc = np.zeros((T)); # learning curve
    obj = np.zeros((T+1)); # initialize objective function curve
    obj[0] = 100; # intial objective value to dummy high value
    losses = np.zeros((T)); # record loss progression over epochs
        
    for ep in range(T):
        np.random.shuffle(idx) # shuffle index
        gt = g0/(1+ep); 
        
        #print('Objective:',obj[ep])    
        for i in idx:
            yi = y[i]; xi = X[i];    
        
            # update weights
            w -= gt*((-yi*xi)*sigmoid(-yi*np.dot(w.T,xi)) + np.divide(2*w,sig2s))            
            
        # logReg objective function
        obj[ep+1] = np.log(1+np.exp(-yi*np.dot(w.T,xi))) + np.divide(np.dot(w.T,w),sig2s)

        #make label predictions & return training accuracy                
        epAcc = accuracy(X,y,w)         
        lc[ep] = epAcc #learning curve
            
        # update results if accuracy improves
        if epAcc > acc0: 
            w_best = w; 
            acc0 = epAcc;
            
        print('-> {:.4f}'.format(epAcc), end=" ")      
        
        # calculat epoch loss
        losses[ep] = loss(X,y,w)
        
        # early stop condition on change in objective over epochs
        if np.abs(obj[ep+1] - obj[ep]) < tau:
            print('\n    Early stop - epoch {}'.format(ep))
            print('    Objective values {:.2f} -> {:.2f}'.format(obj[ep+1], obj[ep]))            
            
            lc = lc[0:ep+1]
            obj = obj[0:ep+1+1]
            losses = losses[0:ep+1]
            
            break                
    
    return w_best, acc0, lc, obj, losses

def sigmoid(z):
    return 1/(1+np.exp(-z))