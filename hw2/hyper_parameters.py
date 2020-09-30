# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:55:37 2020

@author: Alex
"""

# import libraries
import pandas as pd
import numpy as np
import os.path
import itertools
from perceptrons import *

#%% load data for cross-validation

def load_data(path_to_data):

    n_folds = len([f for f in os.listdir(path_to_data)if os.path.isfile(os.path.join(path_to_data, f))])
    data = {}; X = {}; y = {};
    for fold in range(1,n_folds+1):
        path_fold = path_to_data+'fold'+str(fold)+'.csv'
        data[fold] = pd.read_csv(path_fold)
        data[fold].columns = np.arange(0,data[fold].shape[1])
        data[fold] = data[fold].rename(columns={0: 'Label'}) 
        
        y[fold] = data[fold].Label        
        X[fold] = data[fold].drop(['Label'], axis=1)
    
    return data, X, y, n_folds

data, _, _, n_folds = load_data('data/csv-format/CVfolds/')

#%% cross-validation
'''
For this problem we will implement k-folds cross-validation.
'''

def build_CVdata(data,n_folds):
    
    dataCV = {};
    for fold in range(1,n_folds+1):
            
        print('+++ CV Data Fold -', fold,' +++')
        
        dataCV[fold] = {};
        dataCV[fold]['tst'] = data[fold]
                
        # create list of all other folds for training        
        other_folds = list(range(1,6))  
        del other_folds[fold-1]
        
        dataCV[fold]['trn'] = data[other_folds[0]]
       
        del other_folds[0]
        
        for f in other_folds:
            dataCV[fold]['trn'] = dataCV[fold]['trn'].append(data[f], ignore_index=True)

    return dataCV

dataCV = build_CVdata(data,n_folds)

#%% collect hyper-parameters combinations

bestHP = pd.DataFrame(np.zeros((4,5)), index=['std','decay','avg','margin'],
                                    columns=['fold','rate','margin','trn_accuracy', 'test_accuracy'])
weights = {}; biases = {}; 
acc_s0 = acc_d0 = acc_a0 = acc_m0 = 0;
for f in np.arange(1, n_folds+1):
    print('Fold - ', f)
    data_fold = dataCV[f]
    
    # initialize parameters
    np.random.seed(42) # set random seed
    etas = [1, 0.1, 0.01] # learning rates 
    margins = [1, 0.1, 0.01] # margins 
    
    # initialize weights and bias terms
    predAcc_dict = {}; # store accuracies
    w0 = np.random.uniform(-0.01, 0.01, size=(data_fold['trn'].shape[1]-1)) 
    b0 = np.random.uniform(-0.01, 0.01)
    T = 10;
        
#--- standard batch
    for r in etas:
        w_std, b_std = perc_std(data_fold['trn'],w0,b0,r,T)
        trnAcc_s = pred_acc('Standard - Training', data_fold['trn'], w_std, b_std) 
        testAcc_s = pred_acc('Standard - Testing', data_fold['tst'], w_std, b_std) 
    
        if testAcc_s > acc_s0:
            print('-Batch Perceptron - Standard:', r) 
            print('-Update', np.round(acc_s0,3), '->', np.round(testAcc_s,3), f, r, 0)
            acc_s0 = testAcc_s; # if predicted accuracy for these hp is better, update
            bestHP.loc['std'] = [f, r, 0, trnAcc_s, testAcc_s]
            weights['std'] = w_std;
            biases['std'] = b_std;      
                     
# --- decay
#    for r in etas:
    for r in etas:
        w_decay, b_decay = perc_decay(data_fold['trn'],w0,b0,r,T)
        trnAcc_d = pred_acc('Decay - Training', data_fold['trn'], w_decay, b_decay) 
        testAcc_d = pred_acc('Decay - Testing', data_fold['tst'], w_decay, b_decay) 
    
        if testAcc_d > acc_d0:
            print('-Batch Perceptron - Learning Decay:', r) 
            print('-Update', np.round(acc_d0,3), '->', np.round(testAcc_d,3), f, r, 0)
            acc_d0 = testAcc_d; # if predicted accuracy for these hp is better, update
            bestHP.loc['decay'] = [f, r, 0, trnAcc_d, testAcc_d]
            weights['decay'] = w_decay;
            biases['decay'] = b_decay;      
    
# --- average
    for r in etas:
        w_avg, b_avg = perc_avg(data_fold['trn'],w0,b0,r,T)
        trnAcc_a = pred_acc('Decay - Training', data_fold['trn'], w_avg, b_avg) 
        testAcc_a = pred_acc('Decay - Testing', data_fold['tst'], w_avg, b_avg) 
    
        if testAcc_a > acc_a0:
            print('-Batch Perceptron - Averaging:', r)
            print('-Update', np.round(acc_a0,3), '->', np.round(testAcc_a,3), f, r, 0)
            acc_a0 = testAcc_a; # if predicted accuracy for these hp is better, update
            bestHP.loc['avg'] = [f, r, 0, trnAcc_a, testAcc_a]
            weights['avg'] = w_avg;
            biases['avg'] = b_avg;   
    
# --- margin + decay        
    for r, m in list(itertools.product(etas, margins)):
        
        w_margin, b_margin = perc_margin(data_fold['trn'],w0,b0,r,T,m)   
        trnAcc_m = pred_acc('Margin + Decay - Training', data_fold['trn'], w_margin, b_margin) 
        testAcc_m = pred_acc('Margin + Decay - Testing', data_fold['tst'], w_margin, b_margin) 
    
        if testAcc_m > acc_m0:
            print('-Batch Perceptron - Margin + Decay', r, m)   
            print('-Update', np.round(acc_m0,3), '->', np.round(testAcc_m,3), f, r, m)
            acc_m0 = testAcc_m; # if predicted accuracy for these hp is better, update
            bestHP.loc['margin'] = [f, r, m, trnAcc_m, testAcc_m]
            weights['margin'] = w_margin;
            biases['margin'] = b_margin;       

print('Best performance for each perceptron across all folds and parameters')
print(bestHP)
#%%

        
        



























