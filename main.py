"""
    heavy reference taken from:
        https://elitedatascience.com/keras-tutorial-deep-learning-in-python

    todo:
        try model.predict
        figure out the passing in of callbacks
        figure out how to store the network vars somewhere so can just load
        try other datasets
        revise all the terminology and understand what's going on
"""

# import stuff
# normally we'd import from keras directly, but here we import from tensorflow.keras
# to avoid some kind of compatibility issue in the layer- and graph-building code
import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from matplotlib import pyplot as plt # this one allows u to see the mnist data

# constant seed so we can reproduce results
SEED = 0
np.random.seed(SEED)

# load mnist data.
# x -> 28x28 uint8 "grayscale images" of handwritten digits
# y -> labels of those handwritten digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("shape of x_train: " + str(x_train.shape))

# good practice to visually plot the data at the very start,
# to ensure u didn't misinterpret the data dimensions
plt.imshow(x_train[0])

# different backends (tensorflow, theano) expect your channel number (rgb,
# grayscale) to be in different positions within the data.
print("image data format: " + keras_backend.image_data_format())
print("WE ARE ASSUMING TENSORFLOW BACKEND, SO THE ABOVE SHLD BE CHANNELS_LAST")

# reshape the image data to fit ***channels_last***
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print("new x_train shape: " + str(x_train.shape))

# convert all the data elements to float32, and normalize to the range [0,1]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print("x stuff preprocessed (reshaped)\n")


# preprocess the y stuff
# it's given in class vector form. we want it as a binary class matrix.
print("y_train shape: " + str(y_train.shape))
print("first 5 elements: " + str(y_train[:5]))
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print("new y_train shape: " + str(y_train.shape))
print("first 5 elements: " + str(y_train[:5]))
print("y stuff preprocessed\n")


# create layers
layers = []
# input_shape - shape of one sample in dataset.
# first 3 params - num of conv filters, and dimensions of each conv kernel
layers.append(Convolution2D(32, 3, 3, activation="relu", input_shape=(28,28,1)))
layers.append(Convolution2D(32, 3, 3, activation="relu"))
layers.append(MaxPooling2D(pool_size=(2,2)))
layers.append(Dropout(0.25))
layers.append(Flatten())#  weights from Conv MUST be flattened before passing to Dense
layers.append(Dense(128, activation="relu")) # first param - output size
layers.append(Dropout(0.5))
layers.append(Dense(10, activation="softmax"))

model = Sequential(layers)
print("model output shape: " + str(model.output_shape) + "\n")


# compile model
model.compile(
        loss="categorical_crossentropy",    # how "wrong" your output is
        optimizer="adam",                   # how to correct the output
        metrics=["accuracy"])               # judge performance of model

# fit model to data
# epochs: number of iterations thru dataset
# NOTE: can pass in callbacks to access the stages of the training process
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)


# end
score = model.evaluate(x_test, y_test, verbose=1)
print("final score: " + str(score))
