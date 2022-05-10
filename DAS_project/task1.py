from keras.datasets import mnist

data = mnist.load_data()

#Split the dataset
(X_train, y_train), (X_test, y_test) = data

#Reshape the dataset to access every pixel
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')

#Get value of every pixel as a number between 0 and 1
X_train = X_train / 255
X_test = X_test / 255

digit = 4

