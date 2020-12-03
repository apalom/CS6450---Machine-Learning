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

    print('\nLogistic Regression Model')    
    print('[-- HP: lr {} + sig2 {} --]'.format(g0,sig2))
    data_np = data.to_numpy() # split data
    y = data_np[:,0]
    X = data_np[:,1:]
    X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias
    
    w = np.zeros(X.shape[1]); # initialize weights with bias
    acc0 = 0; # initialize accuracy baseline    

    idx = np.arange(X.shape[0]); # index for stepping through data

    lc = np.zeros((T)); # learning curve
    obj = np.zeros((T+1)); # initialize objective function curve
    obj[0] = 100; # intial objective value to dummy high value
    losses = np.zeros((T)); # record loss progression over epochs
        
    for ep in range(T):
        np.random.shuffle(idx) # shuffle index       
        gt = g0/(1+ep) # update learning rate
    
        for i in idx:
            yi = y[i]; xi = X[i];    
        
            # update weights
            w -= gt*((-yi*xi)*sigmoid(-yi*np.dot(w.T,xi)) + np.divide(2*w,sig2))   
                
        # evaluate epoch accuracy, objective, loss
        epAcc, obj[ep+1], losses[ep] = evalEp_LogReg(X,y,w,sig2)
        lc[ep] = epAcc #learning curve
                    
        # update results if accuracy improves
        if epAcc > acc0: 
            w_best = w; 
            acc0 = epAcc;
            better = True;
        
        # print statement
        if better:
            print('-> {:.4f}'.format(epAcc), end=" ")      
        else: print('.', end=" ")      
        better = False;    
           
        if tau != 'None': # do not evaluate for early stop during cross-validation
        # early stop condition on change in objective over epochs
            if ep > 2 and np.abs(obj[ep+1] - obj[ep]) < tau and np.abs(obj[ep] - obj[ep-1]) < tau:
                print('\n    Early stop - epoch {}'.format(ep))
                print('    Objective values {:.3f} -> {:.3f}'.format(obj[ep], obj[ep+1]))            
                
                lc = lc[0:ep+1]
                obj = obj[0:ep+1+1]
                losses = losses[0:ep+1]
                
                break                
    
    return w_best, acc0, lc, obj, losses

def sigmoid(z):
    return 1/(1+np.exp(-z))

#%% make prediction / calculate error

def evalEp_LogReg(X,y,w,sig2):
    
    yi_p = []; acc_cnt = 0;
    # SVM regularizer term
    regularizer = np.divide(np.dot(w.T, w),sig2);  
    
    loss = 0; obj = 0; # initialize loss/obj
    for i in range(X.shape[0]):
        yi = y[i]; xi = X[i];    
        wTdotxi = np.dot(w.T,xi)
        # Log-loss over sample
        loss += np.log(1 + np.exp(-yi*wTdotxi))
        
        # create predicted label
        if wTdotxi >= 0: 
            yi_p.append(1) # true label
        else: yi_p.append(-1) # false label #NOTE check label true/false [1,0] or [1,-1]
    
        # count correct labels
        if yi_p[-1] == yi:
            acc_cnt += 1; 
    
    # calculate accuracy       
    acc = acc_cnt/len(X)                 
    # calculate objective
    obj = regularizer + loss 
    
    return acc, obj, loss