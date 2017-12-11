import numpy as np
import os
from keras.applications import VGG16
from keras.applications import SGD
from keras.preprocessing.image.ImageDataGenerator


#Instantiate a new VGG16 model pretrained with Imagenet weights
model = VGG16(include_top=True, weights='imagenet')

#Print a summary of the model
#model.summary()

# Instantiate and configure the optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#Configure the model for training
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data to test functionality without Imagenet images 
# (These numbers need to be tweaked to match a 224x224 3-channel image)
#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# Augmentation configuration for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Augmentation configuration for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# This is a generator that will read pictures found in
# subfolders of 'data/input/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/input/train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/input/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

# Train the model
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

#Save weights at the end of training 
model.save_weights('first_try.h5')
