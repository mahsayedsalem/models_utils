# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:50:11 2018

@author: mahsayedsalem
"""

from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import os
from random import shuffle
import cv2
from keras.utils.np_utils import to_categorical
import gc
from tqdm import tqdm
import warnings 
warnings.filterwarnings("ignore")

def one_hot(Y_train, Y_test, n_classes):
    
    Y__train_hot = to_categorical(Y_train, num_classes = n_classes)
    Y_test_hot = to_categorical(Y_test, num_classes = n_classes)
    return Y__train_hot, Y_test_hot
    

def under_sampling(X_train, X_test, Y_train, Y_test, height, width, channels, n_classes):
    
    ros = RandomUnderSampler(ratio='auto')
    X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
    X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
    X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
    X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)
    X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
    X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)
    Y_trainRosHot,  Y_testRosHot= one_hot(Y_trainRos, Y_testRos, n_classes)
    
    for i in range(len(X_trainRos)):
        X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)

    for i in range(len(X_testRos)):
        X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)
              
    from sklearn.utils import class_weight
    class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    print("Old Class Weights: ",class_weight)
    from sklearn.utils import class_weight
    class_weight2 = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
    print("New Class Weights: ",class_weight2)  
    return X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot, class_weight2


def memory_management(deleted):
    
    if deleted is not None:
        del deleted
        gc.collect()
    
    
def label_img(Dir_num, num):
    
    labels = []
    for i in range(0,Dir_num):
        labels.append(0)
        
    labels[num] = 1
    return labels

    
def create_train_data_from_folders(Directories, labels, IMG_SIZE, channels, percent):
    
    n_classes = len(Directories)
    full_data = []
    label_decoder = []
    
    for dir in range(len(Directories)):
        label = []
        for img in tqdm(os.listdir(Directories[dir])):
            label = label_img(n_classes, dir)
            path = os.path.join(Directories[dir], img)
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            full_data.append([np.array(img), np.array(label)])
        label_decoder.append([np.array(label), labels[dir]])   
        
    shuffle(full_data)
    train_range = int(len(full_data) * (1-percent))
    train = full_data[:train_range]
    test = full_data[train_range:]
    
    x_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, channels)
    y_train = np.array([i[1] for i in train])
    x_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, channels)
    y_test = np.array([i[1] for i in test])
    np.save('label_decoder.npy', label_decoder)
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)
    return x_train, y_train, x_test, y_test, label_decoder


def normalize(x_train, x_test):
    
    x_train = x_train/255
    x_test = x_test/255
    return x_train, x_test