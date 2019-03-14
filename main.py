#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:42:16 2019

@author: vladgriguta
"""
import gc
gc.collect()
import os


#import os, sys, glob
import readpkl
import plotting

#import multiprocessing
#import time

#ML libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


    
#Function to run randon forest pipeline with feature pruning and analysis
def RF_pipeline(x,name_of_features,y,name_of_classes, train_percent, n_jobs=-1, n_estimators=500,directory='RFres/'):
    
    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_percent,
                                                        random_state=0)
    
    pipeline = Pipeline([ ('RF', RandomForestClassifier(n_jobs=n_jobs,
        n_estimators=n_estimators,random_state=0,class_weight='balanced')) ])
    #do the fit and feature selection
    pipeline.fit(x_train, y_train)
    # check accuracy and other metrics:
    y_pred = pipeline.predict(x_test)
    accuracy=(accuracy_score(y_test, y_pred))
    # Compute the F1 Score
    f1 = f1_score(y_test,y_pred,labels=name_of_classes,average='weighted')

    #make plot of feature importances
    plotting.plot_feature_importance(name_of_features,pipeline,title='Feature Importance RF',
                                     directory=directory)
    
    # Compute and plot the confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plotting.plot_confusion_matrix(cnf_matrix, classes=name_of_classes,directory=directory,
                                   title='Confusion matrix Random Forest')
    return pipeline,y_pred,accuracy,f1



if __name__ == "__main__":
    
    input_table = '../moreData/test_query_table_10k'
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms','subclass','class']

    x,name_of_features,y,name_of_classes,scaler = readpkl.prepare_data(input_table,trim_columns,
                                                                       train_percent=0.7,RF_only=True)
    
    # create new directory if non existent
    directory = 'RFres/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    RFpipeline,_,accuracy_RF,f1_RF = RF_pipeline(x,name_of_features,y,name_of_classes,train_percent=0.7,
                                                 n_jobs=3,directory=directory)
    print('The accuracy of RF is a = '+str(accuracy_RF))
    print('The f1 score of RF is a = '+str(f1_RF))
    
    # RF_pipeline.predict(newX)
    
    
    
    
    
    