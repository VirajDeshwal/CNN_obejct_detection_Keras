#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:15:44 2017

@author: virajdeshwal
"""

from keras.datasets import cifar10
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Dropout, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = plt.figure(figsize= (20,5))
for i in range(36):
    ax =fig.add_subplot(3, 12, i+1, xticks=[], yticks= [])
    ax.imshow(np.squeeze(x_train[i]))
    
    
#one-hot encode the labels 
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# break the training set in training and validation 
(x_train , x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

#print the shape of the training set
print('X_training set is here hurrraaaaayyyyy!!!  : ', x_train.shape)
print('X_training set no. are here hurrraaaaayyyyy!!!  : ', x_train.shape[0])
print('y_training set no. are here hurrraaaaayyyyy!!!  : ', x_test.shape[0])
print('X_validation set no. are here hurrraaaaayyyyy!!!  : ', x_valid.shape[0])

#define the model architecure

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size =2,strides=1, padding = 'same', activation= 'relu' , input_shape =(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters =32, kernel_size=2,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))    
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation = 'softmax'))
model.summary()

model.compile(loss= 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath = 'model.weight.best.hdf5',verbose =1 , save_best_only=True)
hist = model.fit(x_train, y_train, batch_size =32, epochs =20, validation_data=(x_valid, y_valid),
                 callbacks=[checkpointer], verbose =2, shuffle =True)

#load the weights from that yielded best validation accuracy
model.load_weights('model.weight.best.hdf5')
score = model.evaluate(x_test, y_test, verbose=0)
print('\n test accuracy is {}  lol dont laugh'.format(score[1]))

#wooho lets predict the worst prediction of my data 

y_hat = model.predict(x_test)
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
