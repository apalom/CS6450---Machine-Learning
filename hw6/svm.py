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

    print('\nSVM Model')
    print('[-- HP: lr {} + C {} --]'.format(g0,C))
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
        
            if yi*(np.dot(w.T, xi)) <= 1:
                w = (1-gt)*w + gt*C*yi*xi;  
                up+=1                                              
            else: 
                w = (1-gt)*w;                 
            
        # SVM objective function
        obj[ep+1] = 0.5 * np.dot(w.T, w) + C * np.max((0, (1 - yi * np.dot(w.T, xi))))
        
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

#%% make prediction / calculate error

def accuracy(X,y,w):
    
    yi_p = []; acc_cnt = 0;
    
    for i in range(X.shape[0]):
        yi = y[i]; xi = X[i];    
        
        if np.dot(w.T,xi) >= 0: # create predicted label
            yi_p.append(1) # true label
        else: yi_p.append(-1) # false label #NOTE check label true/false [1,0] or [1,-1]
    
        if yi_p[-1] == yi:
            acc_cnt += 1; # count correct labels
           
    acc = acc_cnt/len(X)                 
    
    return acc

def loss(X,y,w):
    
    L = 0
    for i in range(len(X)):
        yi = y[i]; xi = X[i];    
        # SVM loss
        L += np.max((0,1-yi*np.dot(w.T,xi)))
    
    return L















