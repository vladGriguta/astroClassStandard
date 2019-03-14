#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:46:44 2019

@author: vladgriguta
"""

import gc
gc.collect()


#import os, sys, glob
import pandas as pd
import numpy as np

import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))
    
#Function to prepare data for Machine Learning
def prepare_data(filename, trim_columns, train_percent=0.7,RF_only = True):
    
    data_table=load_obj(filename)

    # Drop the entries that do not have a class assigned
    data_table = data_table.replace(np.nan, 'X', regex=True)
    data_table.drop(index=data_table[data_table['class']=='X'].index,inplace=True)
    
    data_table.reset_index(inplace=True)
    
    y=data_table['class']
    name_of_classes = np.array(np.unique(y))
    
    #trim away unwanted columns    
    x=data_table.drop(columns=trim_columns)

    name_of_features = x.columns
    
    # Scale all data
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    if(RF_only == True):
        return x,name_of_features,y,name_of_classes,scaler
    else:
        from keras.utils import np_utils
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        
        # compute weights to account for class imbalace and improve f1 score
        y_cat = np.unique(encoded_Y)
        class_appearences = {y_cat[i]:np.sum(encoded_Y==y_cat[i]) for i in range(len(y_cat))}
        n_classes_norm = len(encoded_Y)/10000
        class_weights = {list(class_appearences.keys())[i]:n_classes_norm/list(class_appearences.values())[i] for i in range(len(class_appearences))}
        
        
        #split data up into test/train
        x_train, x_test, dummy_y_train, dummy_y_test = train_test_split(x,
                        dummy_y, train_size=train_percent, random_state=0)
        
        return x_train, x_test, dummy_y_train, dummy_y_test,encoder,class_weights, name_of_features, name_of_classes