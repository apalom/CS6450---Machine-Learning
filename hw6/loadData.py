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
        print('Reading validation fold:', fold)
        data[fold] = {}
        path_fold = path_to_CV+'fold'+str(fold)+'.csv'
        data[fold]['val'] = pd.read_csv(path_fold, header=None)
        data[fold]['val'].columns = np.arange(0,data[fold]['val'].shape[1])
        data[fold]['val'] = data[fold]['val'].rename(columns={0: 'Label'}) 
        
        # remove validation fold
        trn_folds = np.delete(np.arange(1,n_folds+1),fold-1)
        data[fold]['trn'] = pd.DataFrame()
        for f in trn_folds:            
            path_trnFold = path_to_CV+'fold'+str(f)+'.csv'
            print('   Adding training fold:', path_trnFold)
            data[fold]['trn'] = data[fold]['trn'].append(pd.read_csv(path_trnFold, header=None))
            #pd.read_csv(path_trnFold, header=None)
                
        #data[fold]['trn'].columns = np.arange(0,data[fold]['trn'].shape[1])
        # rename column 0 to label | reset index (remove duplicates from folds)
        data[fold]['trn'] = data[fold]['trn'].rename(columns={0: 'Label'})
        data[fold]['trn'].index = np.arange(dataCV[fold]['trn'].shape[0])
        # reindex training data folds
        #data[fold]['trn'].reindex(np.arange(data[fold]['trn'].shape[1]))
    
    print('\nLoaded {}-fold data'.format(n_folds))
    
    dataTrain = pd.read_csv(path_to_Trn, header=None)
    dataTrain = dataTrain.rename(columns={0: 'Label'}) 
    print('Loaded training data')
    
    dataTest = pd.read_csv(path_to_Tst, header=None)
    dataTest = dataTest.rename(columns={0: 'Label'}) 
    print('Loaded testing data')
    
    return data, dataTrain, dataTest

dataCV, dataTrn, dataTst = load_data('data/csv-format/CVfolds/','data/csv-format/train.csv','data/csv-format/test.csv')

#%% cross-validation data folds for ensemble learning

def ensembleCV(dataTrfm,dataTsfm,depths):
    
    n_folds = 5;
    idx = np.arange(0,len(dataTrfm[1]));
    data = {}; X = {}; y = {};
        
    for maxDepth in depths:
        data[maxDepth] = {}
        np.random.shuffle(idx) # shuffle index
        print('Depth:',maxDepth)
        for fold in range(1,n_folds+1):
            print('-> Fold: {}'.format(fold), end=" ")
            first = (fold-1)*int(0.2*len(idx)); last = first+int(0.2*len(idx));
            idxFold = idx[first:last]
            
            data[maxDepth][fold] = pd.DataFrame(dataTrfm[maxDepth][idxFold])
        print("\n")
    dataTest = 0
    
    return data

dataEnsCV = ensembleCV(dataTrfm,depths)

#%%

idx = np.arange(0,len(dataTrfm[1]))
np.random.shuffle(idx)

for f in n_folds:
    dataf[f]

