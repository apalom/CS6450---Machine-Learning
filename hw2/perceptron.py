# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:20:36 2020

@author: Alex
"""

# import libraries
import pandas as pd
import numpy as np

#%% calculate error

def pred_acc(variant,y,data,w,b):
    wT = w.transpose();
    yi_p = []; acc_cnt = 0;
    
    for ix, row in data.iterrows():
        yi = row.Label # select sample label
        xi = row.drop('Label') # select sample features
        
        if np.dot(wT,xi) + b >= 0: # create predicted label
            yi_p.append(1)
        else: yi_p.append(-1)
    
        if yi_p[-1] == yi:
            acc_cnt += 1;
           
    acc = acc_cnt/len(y)    
    
    print('- Pred accuracy: {:.3f}'.format(acc)) 
    
    return acc

#%% load data

def load_data(path_to_data):
    print('====> Load Data @', path_to_data)
    
    data = pd.read_csv(path_to_data)
    data.columns = np.arange(0,data.shape[1])
    data = data.rename(columns={0: 'Label'}) 
    
    y = data.Label
    X = data.drop(['Label'], axis=1)
    
    return data, X, y

data, X, y = load_data('data/csv-format/train.csv')

#%% initialize parameters
np.random.seed(42) # set random seed
etas = [1, 0.1, 0.01] # learning rates 
margins = [1, 0.1, 0.01] # margins 

# initialize weights and bias terms
predAcc_dict = {}; # store accuracies
w0 = np.random.uniform(-0.01, 0.01, size=(X.shape[1])) 
b0 = np.random.uniform(-0.01, 0.01)
T = 10;

#%% standard batch perceptron algorithm
def perc_std(data,w,b,r,T):      
    
    print('\nBatch Perceptron - Standard')  
    
    wT = w.transpose(); # initialize values
    for ep in range(T):   
        print('.', end=" ")
        data = data.sample(frac=1).reset_index(drop=True)
        
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features
            
            if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                wT = w.transpose()
    
    print('\n- Learning rate:', r)
    
    return w, b

w_std, b_std = perc_std(data,w0,b0,etas[0],T)

predAcc_dict['std'] = pred_acc('Standard',y,data,w_std,b_std)  
        
#%% batch perceptron algorithm with learning decay      
def perc_decay(data,w,b,r,T):   
   
    print('\nBatch Perceptron - Learning Decay')   
    
    wT = w.transpose(); t = 0; r0 = r; # initialize values
    for ep in range(T):   
        print('.', end=" ")
        data = data.sample(frac=1).reset_index(drop=True)
        
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features
            
            if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                wT = w.transpose()
                
            t += 1; # update time step
            r = r0/(1+t); # decay learning rate            
    
    print('\n- Learning rate:', r0, '->', np.round(r,6))
    return w, b

w_decay, b_decay = perc_decay(data,w0,b0,etas[0],T)
#print('Avg weight: {:.3f}'.format(np.average(w)))        
predAcc_dict['decay'] = pred_acc('Learning Decay',y,data,w_decay,b_decay)   

#%% batch perceptron algorithm with averaging with fixed learning rate  
def perc_decay(data,w,b,r,T):   
   
    print('\nBatch Perceptron - Averaging')   
    
    wT = w.transpose(); w_sum = w; b_sum = b; s = 0;# initialize values
    for ep in range(T):   
        print('.', end=" ")
        data = data.sample(frac=1).reset_index(drop=True)
        
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features
            
            if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                wT = w.transpose()
            
            # accumulate weights
            w_sum += w; b_sum += b;
            s += 1;
        
    w_avg = w_sum/s; # average weight
    b_avg = b_sum/s; # average bias    
    
    print('\n- Learning rate:', r) 
    print('- Number of Updates:', s) 
    return w_avg, b_avg

w_avg, b_avg = perc_decay(data,w0,b0,etas[0],T)
#print('Avg weight: {:.3f}'.format(np.average(w)))        
predAcc_dict['avg'] = pred_acc('Averaging',y,data,w_avg,b_avg) 


#%% batch perceptron algorithm with margin and decaying learning rate  
def perc_decay(data,w,b,r,T,margin):   
   
    print('\nBatch Perceptron - Margin + Decay')   
    
    wT = w.transpose(); t = 0; r0 = r; # initialize values
    for ep in range(T):   
        print('.', end=" ")
        data = data.sample(frac=1).reset_index(drop=True)
        
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features
            
            if yi * (np.dot(wT,xi) + b) <= margin: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                wT = w.transpose()
                
            t += 1; # update time step
            r = r0/(1+t); # decay learning rate            
    
    print('\n- Learning rate:', r0, '->', np.round(r,6))
    print('- Margin:', margin)
    return w, b

w_margin, b_margin = perc_decay(data,w0,b0,etas[0],T,margins[0])
   
predAcc_dict['margin'] = pred_acc('Margin + Decay',y,data,w_margin,b_margin)   

#%% 