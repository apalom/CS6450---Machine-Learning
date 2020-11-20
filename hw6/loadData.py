# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:02:23 2020

@author: Alex
"""

#%% load cross-validation and training data

def load_data(path_to_CV, path_to_Trn, path_to_Tst):

    n_folds = len([f for f in os.listdir(path_to_CV)if os.path.isfile(os.path.join(path_to_CV, f))])
    data = {}; X = {}; y = {};
    for fold in range(1,n_folds+1):
        path_fold = path_to_CV+'fold'+str(fold)+'.csv'
        data[fold] = pd.read_csv(path_fold, header=None)
        data[fold].columns = np.arange(0,data[fold].shape[1])
        data[fold] = data[fold].rename(columns={0: 'Label'}) 
    
    print('Loaded {}-fold data'.format(n_folds))
    
    dataTrain = pd.read_csv(path_to_Trn, header=None)
    dataTrain = dataTrain.rename(columns={0: 'Label'}) 
    print('Loaded training data')
    
    dataTest = pd.read_csv(path_to_Tst, header=None)
    dataTest = dataTest.rename(columns={0: 'Label'}) 
    print('Loaded testing data')
    
    return data, dataTrain, dataTest

dataCV, dataTrn, dataTst = load_data('data/csv-format/CVfolds/','data/csv-format/train.csv','data/csv-format/test.csv')
