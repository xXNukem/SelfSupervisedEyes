from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D,MaxPooling2D, Concatenate, BatchNormalization
from keras.optimizers import RMSprop
import dataGenerator
import auxfunctions
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras import regularizers
import tensorflow.keras as k
from keras.constraints import max_norm
from keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas
from keras.callbacks import TensorBoard
import datetime
from sklearn.utils import class_weight
import numpy as np

#------------------------------------------------------------
trainLabels = pandas.read_csv("./trainLabels.csv", dtype=str)

#Hay que añadir la extension a la lista de imagenes
def append_ext(fn):
    return fn+".jpeg"

trainLabels["image"]=trainLabels["image"].apply(append_ext)
#test_data["id_code"]=test_data["id_code"].apply(append_ext)

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    zoom_range=[-2, 2],
    width_shift_range=[-25, 25],
    height_shift_range=[-25, 25],
    rotation_range=40,
    shear_range=40,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.98,1.05],
    featurewise_center=True,
    samplewise_center=True,
    # channel_shift_range=1.5,
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    validation_split=0.25)

mean,std=auxfunctions.getMeanStdClassification()
datagen.mean=mean
datagen.std=std

numClasses = 5
width=240
height=240
input_shape=(width,height,3)

train_generator = datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='./resized_train_cropped',
        x_col="image",
        y_col="level",
        target_size=(240,240),
        batch_size=16,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        subset='training')

validation_generator =datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='./resized_train_cropped',
        x_col="image",
        y_col="level",
        target_size=(240,240),
        batch_size=16,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        subset='validation')

#----------------------------------------------------------------------------------------

def createBaseNetwork(input_shape):
    weight_decay = 1e-4
    L2_norm = regularizers.l2(weight_decay)

    input = Input(shape=input_shape)
    print(input)

    x = Conv2D(96, (9, 9), activation='relu', name='conv1', kernel_regularizer=L2_norm)(input)
    x = MaxPooling2D((3, 3), name='pool1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    x = Conv2D(384, (5, 5), activation='relu', name='conv2', kernel_regularizer=L2_norm)(x)
    x = MaxPooling2D((3, 3), name='pool2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    x = Conv2D(384, (3, 3), activation='relu', name='conv3')(x)
    x = Conv2D(384, (3, 3), activation='relu', name='conv4')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv5')(x)
    x = MaxPooling2D((3, 3), name='pool3')(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)

    return Model(input, x)


# ---------------------------------------------------------------------------------
baseNetwork=createBaseNetwork(input_shape)

for l in baseNetwork.layers:
    l._trainable=False

baseNetwork.load_weights('./ModelWeights7.h5',by_name=True)

input_a = Input(shape=input_shape,name='input1')
outLayers = baseNetwork(input_a)
outLayers = Dense(1024, activation='relu', name='fc2')(outLayers)
outLayers = Dropout(0.2)(outLayers)
outLayers = Dense(512, activation='relu', name='fc3')(outLayers)
outLayers = Dropout(0.2)(outLayers)
outLayers = Dense(256, activation='relu', name='fc4')(outLayers)
outLayers = Dropout(0.2)(outLayers)
outLayers = Dense(128, activation='relu', name='fc5')(outLayers)
outLayers = Dense(5, activation='relu', name='fc6')(outLayers)
classifier = Dense(numClasses, activation='softmax', name='predictions')(outLayers)

model = Model(input_a, classifier)
model.summary()


for layer in baseNetwork.layers:
	print("{}: {}".format(layer, layer.trainable))


optimizer=k.optimizers.Adagrad(learning_rate=1e-3)
model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc'])

parada=callbacks.callbacks.EarlyStopping(monitor='val_acc',mode='max',verbose=1,restore_best_weights=True,patience=3)
learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='val_acc', verbose=1, mode='max',factor=0.2, min_lr=1e-11,patience=3)
log_dir="logs\\fit\\"+'Antes'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes),
            train_generator.classes)

model.fit_generator(generator=train_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=250,
                            epochs=10,
                            validation_steps=250,
                            class_weight=class_weights,
                            callbacks=[tensorboard_callback]
                            )

"""
train_generator.reset()
validation_generator.reset()

for l in baseNetwork.layers:
    l.trainable=True

for layer in baseNetwork.layers:
	print("{}: {}".format(layer, layer.trainable))

optimizer=k.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc'])

log_dir="logs\\fit\\"+'Despues'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit_generator(generator=train_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=250,
                            epochs=10,
                            validation_steps=250,
                            callbacks=[tensorboard_callback]
                            )
"""