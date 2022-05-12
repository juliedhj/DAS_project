from keras.datasets import mnist
import numpy as np
import random
import matplotlib.pyplot as plt


data = mnist.load_data()

#Split the dataset
(X_train, y_train), (X_test, y_test) = data

#Reshape the dataset to access every pixel
#X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
#X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')

#Get value of every pixel as a number between 0 and 1
#X_train = X_train / 255
#X_test = X_test / 255

#1
# Digit = 4
# Digit 4 -> Label 1
# All other digits -> Label -1

y_train = [1 if y==4 else -1 for y in y_train]

#2 
#Shuffle training set randomly
p = np.random.permutation(X_train.shape[0]) 
y_train = np.array(y_train)[p.astype(int)]
X_train = np.array(X_train)[p.astype(int)]

#Split training set in N subsets
def splitSet(n):
    subsets_X = np.array_split(X_train, n)
    subsets_y = np.array_split(y_train, n)
    #print(subsets_X, subsets_y)

splitSet(5)

#3
#print("Label: ", y_train[2])




#
