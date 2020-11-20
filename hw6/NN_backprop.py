# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:30:07 2020

@author: Alex
"""

import numpy as np

X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias

y = np.array([[-1],[1],[1],[-1]])

# weights inputs to hidden neuron z1
wz1 = np.random.uniform(-0.01, 0.01, size=(X.shape[1])) 
# weights inputs to hidden neuron z2
wz2 = np.random.uniform(-0.01, 0.01, size=(X.shape[1])) 
# weights inputs to output y
wo = np.random.uniform(-0.01, 0.01, size=(X.shape[1])) 

def sigmoid(z):
    return 1/(1 + np.exp(-z)) 

def fwdPass(X,y,w1h,w2h,wo):
    
    loss = 0; yp = np.zeros(X.shape[0])
    for i in range(len(X)): # calculate error across each sample
        xi = X[i]; yi = y[i][0]
        z1 = sigmoid(np.dot(wz1.T,xi))
        z2 = sigmoid(np.dot(wz2.T,xi))
        zs = np.array([1,z1,z2])
        yp[i] = sigmoid(np.dot(wo.T,zs))
        
        loss += 0.5*(yi-yp[i])**2
        
    print('Total error: {:.4f}'.format(loss))    
    return loss, yp, zs

loss, yp, zs = fwdPass(X,y,wz1,wz2,wo)

def backProp(X,y,yp,zs,wz1,wz2,wo):

    r = 0.1 # learning rate
    for i in range(len(X)): # calculate error across each sample
        xi = X[i]; yi = y[i][0]; ypi = yp[i];        
        x0 = xi[0]; x1 = xi[1]; x2 = xi[2];
        z0 = zs[0]; z1 = zs[1]; z2 = zs[2]; 
        wo1 = wo[1]; wo2 = wo[2];

        dLdy = yi - ypi # partial derivative L wrt y
        dLdwo0 = dLdy*z0 # partial derivative L wrt bias to output weight
        dLdwo1 = dLdy*z1
        dLdwo2 = dLdy*z2
        
        dz2ds = z2*(1-z2)
        dz1ds = z1*(1-z1)
        dsdwi0 = x0
        dsdwi1 = x1
        dsdwi2 = x2    
        
        dLdwi01 = dLdy*wo1*dz1ds*dsdwi0 # partial derivative L wrt bias from x0 to z1
        dLdwi02 = dLdy*wo2*dz2ds*dsdwi0 # partial derivative L wrt bias from x0 to z2  
        dLdwi11 = dLdy*wo1*dz1ds*dsdwi1 # partial derivative L wrt w11 from x1 to z1
        dLdwi21 = dLdy*wo1*dz1ds*dsdwi2 # partial derivative L wrt w21 from x2 to z1
        dLdwi12 = dLdy*wo2*dz2ds*dsdwi1 # partial derivative L wrt w12 from x1 to z2
        dLdwi22 = dLdy*wo2*dz2ds*dsdwi2 # partial derivative L wrt w22 from x2 to z2

        # weight updates
        wo -= r*np.array([dLdwo0, dLdwo1, dLdwo2])        
        wz1 -= r*np.array([dLdwi01, dLdwi11, dLdwi21])
        wz2 -= r*np.array([dLdwi02, dLdwi12, dLdwi22])
        
    return wo, wz1, wz2

#wo, wz1, wz2 = backProp(X,y,yp,zs,wz1,wz2,wo)