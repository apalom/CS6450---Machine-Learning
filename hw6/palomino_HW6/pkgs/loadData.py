# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:02:23 2020

@author: Alex
"""

# import libraries
import time as time
import pandas as pd
import numpy as np
np.random.seed(42)
import os
import os.path
import itertools
import matplotlib.pyplot as plt

#%% load cross-validation and training data

def loadData(path_to_CV, path_to_Trn, path_to_Tst):

    n_folds = len([f for f in os.listdir(path_to_CV)if os.path.isfile(os.path.join(path_to_CV, f))])
    data = {}; X = {}; y = {};
    for fold in range(1,n_folds+1):
        print('Reading validation fold:', fold)
        data[fold] = {}
        path_fold = path_to_CV+'fold'+str(fold)+'.csv'
        data[fold]['val'] = pd.read_csv(path_fold, header=None)
        data[fold]['val'].columns = np.arange(0,data[fold]['val'].shape[1])
        data[fold]['val'] = data[fold]['val'].rename(columns={0: 'Label'}) 
        
        # remove validation fold
        trn_folds = np.delete(np.arange(1,n_folds+1),fold-1)
        data[fold]['trn'] = pd.DataFrame()
        # read in training folds
        for f in trn_folds:            
            path_trnFold = path_to_CV+'fold'+str(f)+'.csv'
            print('   Adding training fold:', path_trnFold)
            data[fold]['trn'] = data[fold]['trn'].append(pd.read_csv(path_trnFold, header=None))

        # rename column 0 to label | reset index (remove duplicates from folds)
        data[fold]['trn'] = data[fold]['trn'].rename(columns={0: 'Label'})
        data[fold]['trn'].index = np.arange(data[fold]['trn'].shape[0])
    
    print('\nLoaded {}-fold data'.format(n_folds))
    
    dataTrain = pd.read_csv(path_to_Trn, header=None)
    dataTrain = dataTrain.rename(columns={0: 'Label'}) 
    print('Loaded training data')
    
    dataTest = pd.read_csv(path_to_Tst, header=None)
    dataTest = dataTest.rename(columns={0: 'Label'}) 
    print('Loaded testing data')
    
    return data, dataTrain, dataTest
