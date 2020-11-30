# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:56:58 2020

@author: Alex
"""

# import libraries
import pandas as pd
import numpy as np
import time as time
from results import *

#%% logReg model

def logReg(data, g0, sig2, tau, T):

    print('Logistic Regression Model')    
    print('[-- HP: lr {} + sig2 {} --]'.format(g0,sig2))
    data_np = data.to_numpy() # split data
    y = data_np[:,0]
    X = data_np[:,1:]
    X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias
    
    w = np.zeros(X.shape[1]); # initialize weights with bias
    acc0 = 0; # initialize accuracy baseline    
    up = 0; # initialize update counters
    idx = np.arange(X.shape[0]); # index for stepping through data
    gt = g0; # learning rate
    lc = np.zeros((T)); # learning curve
    obj = np.zeros((T+1)); # initialize objective function curve
    obj[0] = 10**5; # intial objective value to dummy high value
    losses = np.zeros((T)); # record loss progression over epochs
        
    for ep in range(T):
        np.random.shuffle(idx) # shuffle index       
 
        for i in idx:
            yi = y[i]; xi = X[i];    
        
            # update weights
            w = w - gt*((-yi*xi)*sigmoid(-yi*np.dot(w.T,xi)) + np.divide(2*w,sig2))   
            
        # logReg objective function    
        obj[ep+1] = np.log(1+np.exp(-yi*np.dot(w.T,xi))) + np.divide(np.dot(w.T,w),sig2)

        #make label predictions & return training accuracy                
        epAcc = accuracy(X,y,w)         
        lc[ep] = epAcc #learning curve
            
        # update results if accuracy improves
        if epAcc > acc0: 
            w_best = w; 
            acc0 = epAcc;
            
        print('-> {:.4f}'.format(epAcc), end=" ")      
        
        # calculat epoch loss
        losses[ep] = lossLogReg(X,y,w)
        
        # early stop condition on change in objective over epochs
        if np.abs(obj[ep+1] - obj[ep]) < tau:
            print('\n    Early stop - epoch {}'.format(ep))
            print('    Objective values {:.3f} -> {:.3f}'.format(obj[ep], obj[ep+1]))            
            
            lc = lc[0:ep+1]
            obj = obj[0:ep+1+1]
            losses = losses[0:ep+1]
            
            break                
    
    return w_best, acc0, lc, obj, losses

def sigmoid(z):
    return 1/(1+np.exp(-z))

def lossLogReg(X,y,w):
    
    L = 0
    for i in range(len(X)):
        yi = y[i]; xi = X[i];    
        # Logreg loss
        L += np.log(1 + np.exp(-yi * np.dot(w.T,xi)))
    
    return L