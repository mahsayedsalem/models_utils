# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:38:31 2018

@author: mahsayedsalem
"""
import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from scipy.misc import imresize, imread
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json, Model, load_model
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Input, Add, AveragePooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import warnings 
warnings.filterwarnings("ignore")



'''
#######################################################################
The first section is dedicated to functions responsible of 
plotting learning curves and confusion matrices.

## No Args needed, they are already embedded in the networks. 

#######################################################################
'''


class MetricsCheckpoint(Callback):
    
    def __init__(self, savepath):
        
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
        
        
    def on_epoch_end(self, epoch, logs=None):
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


def plotKerasLearningCurve(checkpointslog):
    
    plt.figure(figsize=(10,5))
    metrics = np.load(checkpointslog)[()]
    filt = ['acc']
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')
    

def plot_accuracy_reports(history, log, y_true, y_pred_classes, labels):
    plotKerasLearningCurve(log)
    plt.show()
    plot_learning_curve(history)
    plt.show()
    confusion_mtx = confusion_matrix(y_true, y_pred_classes, labels) 
    plot_confusion_matrix(confusion_mtx, classes = list(labels.values())) 
    plt.show()
    

'''
#######################################################################
Data Generator
# No Args needed, they are already embedded in the networks. 
## You can change the variables values when needed. The default vales
## are the most used in most projects.
#######################################################################
'''


def data_denerator():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    return datagen


'''
#######################################################################

Classification Networks

# Args:
       1) Training Photos
       2) Training Photos Labels
       3) Test Photos
       4) Test Photos Labels
       5) Number of classes in your dataset
       6) Batch Size
       7) Number of epochs
       8,9,10) Image Size(rows, columns, channels)
       11) The activation layer in the hidden layers, should be parsed as string
       12) The activation layer in the output layer, should be parsed as string
       13) The labels in your dataset (Should be a dictionary, each one-hot value corresponds to a string of the label)
       14) The name of the log which will be saved and used to be plotted. Use a unique string
       15) class weights. can be obtained using the following code:
        from sklearn.utils import class_weight
        class_weight2 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
       16) A unique string with extension '.h5' which the model will be saved by, then you can load the model directly later. 
        
# Output:
        1) A test accuracy print
        2) Classification Report
        3) A line plot of both the validation and train sets accuracy behavior throughout the training
        4) 2 plots, the first is like the previos, the second is for loss behavior throughout the training
        5) A confusion matrix(The number of false-postives, and false-negatives, etc)
        6) A saved model of extention .h5 which you can load later and predict using it without having to go through the train phase again
        ## To load the model you can use the following codes ( assume while training you parsed the model_unique_name = 'Mahmoud.h5')
        ## When loading :
            from keras.models import model_from_json
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("Mahmoud.h5")
            print("Loaded model from disk")
            
#Comments: 1)I've not tested the ResNet on RGB images. Only Greyscale ones (number of channels to be one). 
             I will try to test it later, and if anyone can test it feel free to pull request the result.
           2) Feel free to pull request other networks that you tried and worked fine for you and reference the paper on which it's based on. I can be added. 
              Just remember to add the network using the same pipeline of arguments and outputs.
           

#Networks:
        1) Not a famous network, just the one gave the best results for my graduation project
        2) LeNet
        3) ResNet
        
#######################################################################
'''


def classify_network_1(x_train, 
                       y_train, 
                       x_test, 
                       y_test, 
                       num_classes, 
                       batch_size, 
                       epochs, 
                       img_rows, 
                       img_cols, 
                       channels, 
                       middle_layers_activation, 
                       last_layer_activation,
                       labels,
                       checkpointslog,
                       class_weight,
                       model_unique_name):
    
    input_shape = (img_rows, img_cols, channels)
    model = Sequential()
    model.add(Convolution2D(12, 5, 5, activation = middle_layers_activation, input_shape=input_shape, init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(25, 5, 5, activation = middle_layers_activation, init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(180, activation = middle_layers_activation, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = middle_layers_activation, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=last_layer_activation))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    datagen = data_denerator()
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs,class_weight=class_weight, shuffle=True, validation_data = [x_test, y_test], callbacks = [MetricsCheckpoint(checkpointslog)])
    score = model.evaluate(x_test,y_test, verbose=0)
    print('\n classify_network_1 - accuracy:', score[1],'\n')
    y_pred = model.predict(x_test)
    print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='')    
    Y_pred_classes = np.argmax(y_pred,axis=1) 
    Y_true = np.argmax(y_test,axis=1) 
    plotKerasLearningCurve(checkpointslog)
    plt.show()  
    plot_learning_curve(history)
    plt.show()
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = list(labels.values())) 
    plt.show()
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_unique_name)
    print("Saved model to disk")
    
      
def classify_network_2(x_train, 
                       y_train, 
                       x_test, 
                       y_test, 
                       num_classes, 
                       batch_size, 
                       epochs, 
                       img_rows, 
                       img_cols, 
                       channels, 
                       middle_layers_activation, 
                       last_layer_activation,
                       labels,
                       checkpointslog,
                       class_weight,
                       model_unique_name):
    
    INIT_LR = 1e-3
    input_shape = (img_rows, img_cols, channels)
    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding="same",
            input_shape=input_shape))
    model.add(Activation(middle_layers_activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation(middle_layers_activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation(middle_layers_activation)) 
    model.add(Dense(num_classes))
    model.add(Activation(last_layer_activation))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
    datagen = data_denerator()
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs,class_weight=class_weight, shuffle=True, validation_data = [x_test, y_test], callbacks = [MetricsCheckpoint(checkpointslog)])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n classify_network_2 - accuracy:', score[1],'\n')
    y_pred = model.predict(x_test)
    print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='')    
    Y_pred_classes = np.argmax(y_pred,axis=1) 
    Y_true = np.argmax(y_test,axis=1) 
    plotKerasLearningCurve(checkpointslog)
    plt.show()  
    plot_learning_curve(history)
    plt.show()
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = list(labels.values())) 
    plt.show()
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_unique_name)
    print("Saved model to disk")


'''
#######################################################################
ResNet
#######################################################################
'''


def identity_block(X, f, filters, stage, block, middle_layers_activation):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation(middle_layers_activation)(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation(middle_layers_activation)(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Add()([X, X_shortcut])
    X = Activation(middle_layers_activation)(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2, middle_layers_activation):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation(middle_layers_activation)(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation(middle_layers_activation)(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation(middle_layers_activation)(X)

    return X


def ResNet(x_train, 
           y_train, 
           x_test, 
           y_test, 
           num_classes, 
           batch_size, 
           epochs, 
           img_rows, 
           img_cols, 
           channels, 
           middle_layers_activation, 
           last_layer_activation,
           labels,
           checkpointslog,
           class_weight,
           model_unique_name):
    
    input_shape = (img_rows, img_cols, channels)
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(img_rows, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation(middle_layers_activation)(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[img_rows, img_rows, 256], stage=2, block='a', s=1, middle_layers_activation)
    X = identity_block(X, 3, [img_rows, img_rows, 256], stage=2, block='b', middle_layers_activation)
    X = identity_block(X, 3, [img_rows, img_rows, 256], stage=2, block='c', middle_layers_activation)


    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    X = Flatten()(X)
    X = Dense(num_classes, activation=last_layer_activation, name='fc' + str(num_classes), kernel_initializer=glorot_uniform(seed=0))(X)
   
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    datagen = data_denerator()
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs,class_weight=class_weight, shuffle=True, validation_data = [x_test, y_test], callbacks = [MetricsCheckpoint(checkpointslog)])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n ResNet - accuracy:', score[1],'\n')
    y_pred = model.predict(x_test)
    print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='')   
    Y_pred_classes = np.argmax(y_pred,axis=1)
    Y_true = np.argmax(y_test,axis=1) 
    plotKerasLearningCurve(checkpointslog)
    plt.show()
    plot_learning_curve(history)
    plt.show()
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(confusion_mtx, classes = list(labels.values()))
    plt.show()
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_unique_name)
    print("Saved model to disk")
    
####################################################################################
    
    
    
    
'''
#######################################################################
Using Pre-trained Models
#######################################################################
'''

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input


'''
MODELS  = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception,
	"resnet": ResNet50
}
'''
 

def preprocess_image_pretrained(inputs, network):
    
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input
    img_output = []
    
    if network in ("inception", "xception"):
    	inputShape = (299, 299)
    	preprocess = preprocess_input
        
    for img in inputs:
        image = load_img(img, target_size=inputShape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess(image)
        img_output.append(image)
        
    return img_out


def vgg_16_pretrained(inputs):
    
    Network = VGG16
    model = Network(weights="imagenet")
    img_input = preprocess_image_pretrained(inputs, 'vgg16')
    predictions = []
    
    for img in img_input:
        pres = mode.predict(img)
        P = imagenet_utils.decode_predictions(preds)
        (imagenetID, label, prob) = P[0][0]
        predictions.append([np.array(img), label])
        orig = cv2.imread(inputs[img])
        cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)
    
    return predictions


def vgg_19_pretrained(inputs):
    
    Network = VGG19
    model = Network(weights="imagenet")
    img_input = preprocess_image_pretrained(inputs, 'vgg19')
    predictions = []
    
    for img in img_input:
        pres = mode.predict(img)
        P = imagenet_utils.decode_predictions(preds)
        (imagenetID, label, prob) = P[0][0]
        predictions.append([np.array(img), label])
        orig = cv2.imread(inputs[img])
        cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)
    
    return predictions


def inception_pretrained(inputs):
    
    Network = InceptionV3
    model = Network(weights="imagenet")
    img_input = preprocess_image_pretrained(inputs, 'inception')
    predictions = []
    
    for img in img_input:
        pres = mode.predict(img)
        P = imagenet_utils.decode_predictions(preds)
        (imagenetID, label, prob) = P[0][0]
        predictions.append([np.array(img), label])
        orig = cv2.imread(inputs[img])
        cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)
    
    return predictions


def xception_pretrained(inputs):
    
    Network = Xception
    model = Network(weights="imagenet")
    img_input = preprocess_image_pretrained(inputs, 'xception')
    predictions = []
    
    for img in img_input:
        pres = mode.predict(img)
        P = imagenet_utils.decode_predictions(preds)
        (imagenetID, label, prob) = P[0][0]
        predictions.append([np.array(img), label])
        orig = cv2.imread(inputs[img])
        cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)
    
    return predictions


def resnet_pretrained(inputs):
    
    Network = ResNet50
    model = Network(weights="imagenet")
    img_input = preprocess_image_pretrained(inputs, 'resnet')
    predictions = []
    
    for img in img_input:
        pres = mode.predict(img)
        P = imagenet_utils.decode_predictions(preds)
        (imagenetID, label, prob) = P[0][0]
        predictions.append([np.array(img), label])
        orig = cv2.imread(inputs[img])
        cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)























