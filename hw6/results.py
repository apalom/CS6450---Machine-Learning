# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:56:23 2020

@author: Alex
"""

import matplotlib.pyplot as plt
import numpy as np

#%% accuracy

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

#%% plot objective curves

def plot_learning(lc, obj, lr, C, tau, figName):
    # set plotting parameters
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['legend.fontsize'] = 12
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
    
    ax.plot(lc, label='Learning')
    
    ax2 = ax.twinx()
    obj[0] = 0#obj[1]
    ax2.plot(obj, c='grey', linestyle='--', label='Obj')
    
    ax.set_title('Results lr {:.3f}, C {}, tau {}'.format(lr, C, tau))
    ax.set_xticks(np.arange(0,len(lc),2))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Objective')
    fig.legend(bbox_to_anchor=(0.26, 0.88)) 
    
    figPath = 'figs\\'+figName
    fig.savefig(figPath, format='pdf', dpi=300) 
    
    return

#%% plot loss curve

def plot_loss(loss, lr, C, tau, figName):
    # set plotting parameters
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['legend.fontsize'] = 12
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=100)   
    
    ax.plot(loss, c='r')
        
    ax.set_title('Results lr {:.3f}, C {}, tau {}'.format(lr, C, tau))
    ax.set_xticks(np.arange(0,len(loss)+1,2))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cummulative Loss')
    
    fig.tight_layout()
    figPath = 'figs\\'+figName
    fig.savefig(figPath, format='pdf', dpi=300) 
    
    return
