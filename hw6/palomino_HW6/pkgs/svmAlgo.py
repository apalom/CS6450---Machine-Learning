# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:22:24 2020

@author: Alex
"""

# import libraries
import pandas as pd
import numpy as np

#%% SVM model

def svm(data, g0, C, tau, T):

    #print('\nSVM Model')
    #print('[-- HP: lr {} + C {} --]'.format(g0,C))
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
        gt = g0/(1+ep);         
    
        # update weights
        for i in idx:
            yi = y[i]; xi = X[i];    
        
            if yi*(np.dot(w.T, xi)) <= 1:
                w = (1-gt)*w + gt*C*yi*xi;                                          
            else: 
                w = (1-gt)*w;                 
        
        # evaluate epoch accuracy, objective, loss
        epAcc, obj[ep+1], losses[ep] = evalEp_SVM(X,y,w,C)
        lc[ep] = epAcc #learning curve
            
        # update results if accuracy improves
        if epAcc > acc0: 
            w_best = w; 
            acc0 = epAcc;
            better = True;        

        # print statement
        #if better:
        #    print('-> {:.4f}'.format(epAcc), end=" ")      
        #else: print('.', end=" ")      
        #better = False;    
           
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

#%% make prediction / calculate error

def evalEp_SVM(X,y,w,C):
    
    yi_p = []; acc_cnt = 0;
    # SVM regularizer term
    regularizer = 0.5*np.dot(w.T, w);  
    loss = 0; obj = 0; # initialize loss/obj
    for i in range(X.shape[0]):
        yi = y[i]; xi = X[i];    
        wTdotxi = np.dot(w.T,xi)
        # SVM loss over sample
        loss += max(0.0, 1.0 - yi*wTdotxi )
        
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
    obj = regularizer + C * loss 
    
    return acc, obj, loss

