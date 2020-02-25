# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN
import keras
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN

classifier = Sequential()

# Step 1 - Convolution

classifier.add(Conv2D(32, kernel_size = (3, 3) , input_shape =(64, 64, 3), activation = 'relu', data_format = 'channels_last'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# another convolutional layer
classifier.add(Conv2D(32, kernel_size = (3, 3) ,  activation = 'relu', data_format = 'channels_last'))

# max pooling applied to 2nd layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu' )) # hidden layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',  activation = 'sigmoid' )) # output layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(                                                                  
            'dataset/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=2,
        validation_data=test_set,
        validation_steps=2000)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/sinamahbobi/Desktop/Udemy Machine Learning/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set/cats/cat.4007.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
 prediction = 'dog'

else:
 prediction = 'cat'
print(prediction)