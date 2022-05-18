from keras.datasets import mnist
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical


data = mnist.load_data()

#Split the dataset
(X_train, y_train), (X_test, y_test) = data


#Reshape the dataset to access every pixel
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')

#Get value of every pixel as a number between 0 and 1
X_train = X_train / 255
X_test = X_test / 255

#X_train = [..., None]
#Y_train = [..., None]

#1
# Digit = 4
# Digit 4 -> Label 1
# All other digits -> Label -1

y_train = [1 if y==4 else -1 for y in y_train]

#2 
#Shuffle training set randomly
p = np.random.permutation(len(X_train)) 
y_train = np.array(y_train)[p.astype(int)]
X_train = np.array(X_train)[p.astype(int)]

#Split training set in N subsets
def splitSet(n):
    subsets_X = np.array_split(X_train, n)
    subsets_y = np.array_split(y_train, n)
    return subsets_X, subsets_y


#3 Gradient Tracking Algorithm -> distributed algorithm because we only have to exchange local data
network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(28 * 28)))
network.add(Dense(2, activation='softmax'))

network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

def trainSet(n):
    #n is number of agents
    (subsets_x, subsets_y) = splitSet(n)
    for i in range(0,n):
        #network.fit(np.asarray(subsets_x[i]), np.asarray(subsets_y[i]), epochs=10, batch_size=100)
        network.fit(subsets_x[i], subsets_y[i], epochs=5, batch_size=128)
trainSet(5)


