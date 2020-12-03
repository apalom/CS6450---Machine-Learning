# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 22:58:18 2020

@author: Alex
"""


#%% check trees

checkTress = {}
for d in [4]:
    print('--- Depth:',d)
    for t in trees[d]: 
        print('\n',trees[d][t])        


#%% count nans in transformed data
cnt_nans = {}
print('Training Data')
for d in depths:
    cnt_nans[d] = np.round(np.count_nonzero(np.isnan(dataTrfm_trn[d]))/dataTrfm_trn[d].size,4)

print(cnt_nans)

#% count nans in transformed CV data
print('CV Data')
cnt_nans={}
for f in dataTrfm_CV:    
    for d in depths:        
        dataTrfm_CVfoldD = dataTrfm_CV[f]['trn'][d]
        cnt_nans[f,d] = np.round(np.count_nonzero(np.isnan(dataTrfm_CVfoldD))/dataTrfm_CVfoldD.size,5)
print(cnt_nans)
    
#%% replace negatives with zero
# note: worsened performance!

X0 = np.where(X<0, 0, X)
df[df < 0] = 0
