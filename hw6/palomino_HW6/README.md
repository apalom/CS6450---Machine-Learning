## ABOUT
This Jupyter Notebook includes HW# 6 code and data (Alejandro Palomino). The implementation of SVM, Logistic Regression, and SVM over Trees is discussed in the accompanying HW# 6 technical report. This README details the steps necessary to execute the three learning experiments.

### Directory Summary
- data: directory for cross-validation, training, testing data (specifically 'data/'csv-format/')
- figs: output directory for plots
- pkgs: directory for algorithm and support packages
1_mainSVM.pynb: executes SVM learning
2_mainLogReg.pynb: executes Logistic Regression learning
3_mainSVMoverTrees.pynb: executes SVM over Trees learning
CS_64350__Machine_Learning_HW_6_Palomino.pdf: technical report
README.md: This file

### Packages
The core learning algorithms (SVM, Logistic Regression, ID3) are packaged as .py files in the '/pkgs' directory of this Jupyter Notebook. Additionally, there is a package to load data (loadData.py) and plot results (results.py). The 'main' files at the top-level of the directory call upon these packages to complete the experiments. 

### SVM
The **1_mainSVM.pynb** can be exuected from top-to-bottom in the Jupyter environment. In order, the cells import libraries/packages, load data, run SVM cross-validation (with the help of the svmAlgo.py package), train the model with the best hyper-parameters, plot training loss, and finally evaluate the model on the test data. The best-hyper parameters are hard-coded in the training cell, after taking the mode of cross-validation results.

### Logistic Regression
The **2_mainLogReg.pynb** can be exuected from top-to-bottom in the Jupyter environment. The Logistic Regression workflow follows the same as SVM (but relies upon the logRegalgo.py module for learning).

### SVM Over Trees
The **3_mainSVMoverTrees.pynb** can be exuected from top-to-bottom in the Jupyter environment. In order the cells, the cells import libraries/packages, load data, build decision trees (with the help of the id3Algo.py package), transforms loaded data based on the decisoin trees, and then continues in the same manner as 1_mainSVM.pynb.