import numpy as np
np.random.seed(1)
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten

model=Sequential()

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

#3rd convolution layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

#4th convolution layer
model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1), strides=(2, 2)))

#5th convolution layer
model.add(Conv2D(1024, (2, 2), activation='relu'))
model.add(Conv2D(1024, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1), strides=(2, 2)))

# converting to 1-d array
model.add(Flatten())

# add 3 dense layers with dropouts
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))

# output

model.add(Dense(7, activation='softmax')) # check if there were 7 expressions)
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))


# compilation 


# Fit the model
