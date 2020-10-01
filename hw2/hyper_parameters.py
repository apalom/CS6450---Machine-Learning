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

bestHP = pd.DataFrame(np.zeros((4,6)), index=['std','decay','avg','margin'],
                                    columns=['fold','rate','margin','updates','trn_accuracy', 'test_accuracy'])
weights = {}; biases = {}; 
acc_s0 = acc_d0 = acc_a0 = acc_m0 = 0; # initialize baseline accuracy value
up_s0 = up_d0 = up_a0 = up_m0 = 0; # initialize update counts 
for f in np.arange(1, n_folds+1):
    up_s0 = up_d0 = up_a0 = up_m0 = 0; # initialize updates counter for each fold
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
    print('<<< Standard Perceptron >>>')
    for r in etas:        
        w_std, b_std, _, _, _, _ = perc_std(data_fold['trn'],w0,b0,r,T)
        trnAcc_s = pred_acc('Standard - Training', data_fold['trn'], w_std, b_std) 
        testAcc_s = pred_acc('Standard - Testing', data_fold['tst'], w_std, b_std) 
    
        if testAcc_s > acc_s0:
            print('\n-Batch Perceptron - Standard:', r) 
            print('-Update', np.round(acc_s0,3), '->', np.round(testAcc_s,3), f, r, 0)
            acc_s0 = testAcc_s; # if predicted accuracy for these hp is better, update
            up_s0 += 1;
            bestHP.loc['std'] = [f, r, 0, up_s0, trnAcc_s, testAcc_s]
            weights['std'] = w_std;
            biases['std'] = b_std;      
                     
# --- decay
#    for r in etas:
    print('<<< Decay Perceptron >>>')
    for r in etas:
        w_decay, b_decay, _, _, _, _ = perc_decay(data_fold['trn'],w0,b0,r,T)
        trnAcc_d = pred_acc('Decay - Training', data_fold['trn'], w_decay, b_decay) 
        testAcc_d = pred_acc('Decay - Testing', data_fold['tst'], w_decay, b_decay) 
    
        if testAcc_d > acc_d0:
            print('\n-Batch Perceptron - Learning Decay:', r) 
            print('-Update', np.round(acc_d0,3), '->', np.round(testAcc_d,3), f, r, 0)
            acc_d0 = testAcc_d; # if predicted accuracy for these hp is better, update
            up_d0 += 1;
            bestHP.loc['decay'] = [f, r, 0, up_d0, trnAcc_d, testAcc_d]
            weights['decay'] = w_decay;
            biases['decay'] = b_decay;      
    
# --- average
    print('<<< Averaging Perceptron >>>')
    for r in etas:
        w_avg, b_avg, _, _, _, _ = perc_avg(data_fold['trn'],w0,b0,r,T)
        trnAcc_a = pred_acc('Decay - Training', data_fold['trn'], w_avg, b_avg) 
        testAcc_a = pred_acc('Decay - Testing', data_fold['tst'], w_avg, b_avg) 
    
        if testAcc_a > acc_a0:
            print('\n-Batch Perceptron - Averaging:', r)
            print('-Update', np.round(acc_a0,3), '->', np.round(testAcc_a,3), f, r, 0)
            acc_a0 = testAcc_a; # if predicted accuracy for these hp is better, update
            up_a0 += 1;
            bestHP.loc['avg'] = [f, r, 0, up_a0, trnAcc_a, testAcc_a]
            weights['avg'] = w_avg;
            biases['avg'] = b_avg;   
    
# --- margin + decay   
    print('<<< Margin + Decay Perceptron >>>')     
    for r, m in list(itertools.product(etas, margins)):
        
        w_margin, b_margin, _, _, _, _ = perc_margin(data_fold['trn'],w0,b0,r,T,m)   
        trnAcc_m = pred_acc('Margin + Decay - Training', data_fold['trn'], w_margin, b_margin) 
        testAcc_m = pred_acc('Margin + Decay - Testing', data_fold['tst'], w_margin, b_margin) 
    
        if testAcc_m > acc_m0:
            print('\n-Batch Perceptron - Margin + Decay', r, m)   
            print('-Update', np.round(acc_m0,3), '->', np.round(testAcc_m,3), f, r, m)
            acc_m0 = testAcc_m; # if predicted accuracy for these hp is better, update
            up_m0 += 1;
            bestHP.loc['margin'] = [f, r, m, up_m0, trnAcc_m, testAcc_m]
            weights['margin'] = w_margin;
            biases['margin'] = b_margin;       

print('Best performance for each perceptron across all CV folds and parameters.')
print(bestHP)
#%%

# get training data
train, _, _ = load_trainData('data/csv-format/train.csv')

# initialize parameters
np.random.seed(42) # set random seed

# initialize weights and bias terms
predAcc20 = pd.DataFrame(np.zeros((4,3)),index=['std','decay','avg','margin'],
                         columns=['epoch','accuracy','updates']) # store accuracies
lc = {}; up = {};
w0 = np.random.uniform(-0.01, 0.01, size=(data.shape[1]-1)) 
b0 = np.random.uniform(-0.01, 0.01)
T = 20;

# standard perceptron with CV best parameters
print('std')
r_std = bestHP.loc['std'].rate
w_std, b_std, predAcc20.loc['std'], lc['std'], w_stdEp, b_stdEp = perc_std(train,w0,b0,r_std,T)

# decay perceptron with CV best parameters
print('decay')
r_decay = bestHP.loc['decay'].rate
w_decay, b_decay, predAcc20.loc['decay'], lc['decay'], w_decayEp, b_decayEp = perc_decay(train,w0,b0,r_decay,T)  

# average perceptron with CV best parameters
print('avg')
r_avg = bestHP.loc['avg'].rate
w_avg, b_avg, predAcc20.loc['avg'], lc['avg'], w_avgEp, b_avgEp = perc_avg(train,w0,b0,r_avg,T)

# margin perceptron with CV best parameters
print('margin')
r_margin = bestHP.loc['margin'].rate
m_margin = bestHP.loc['margin'].margin
w_margin, b_margin, predAcc20.loc['margin'], lc['margin'], w_marginEp, b_marginEp = perc_margin(train,w0,b0,r_margin,T,m_margin)   

print('Best performance and epoch over 20 epoch training.')
print(predAcc20)

#%% Plot learning curves

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"
variants = ['std','decay','avg','margin']
for v in variants:
    plt.plot(lc[v], label=v)

plt.legend()
plt.xlim(0,20)
plt.xticks(np.arange(0,20,2))    
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Perceptron Learning Curves')

#%% evaluate on test set
        
test, _, _ = load_trainData('data/csv-format/test.csv')

testAcc = {};
testAcc['std'] = pred_acc('Standard - Testing', test, w_stdEp, b_stdEp) 
testAcc['decay'] = pred_acc('Decay - Testing', test, w_decayEp, b_decayEp) 
testAcc['avg'] = pred_acc('Average - Testing', test, w_avgEp, b_avgEp) 
testAcc['margin'] = pred_acc('Margin + Decay - Testing', test, w_marginEp, b_marginEp) 

print('Test accuracy for each perceptron.')
for v in variants:
    print(v,"-",np.round(testAcc[v],4))
















