from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D,MaxPooling2D, Concatenate, BatchNormalization
from keras.optimizers import RMSprop
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras import regularizers
import tensorflow.keras as k
from keras.constraints import max_norm
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas
from tensorflow.keras.callbacks import TensorBoard
import datetime
from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import pickle
import os
#------------------------------------------------------------

trainLabels = pandas.read_csv("../data/trainLabels.csv", dtype=str)

#Hay que añadir la extension a la lista de imagenes
def append_ext(fn):
    return fn+".jpeg"

trainLabels["image"]=trainLabels["image"].apply(append_ext)
#test_data["id_code"]=test_data["id_code"].apply(append_ext)


datagen = ImageDataGenerator(
    height_shift_range=[-1.5, 1.5],
    width_shift_range=[-1.5, 1.5],
    shear_range=2,
    zoom_range=[-1.5, 1.5],
    rotation_range=2,
    channel_shift_range=1.5,
    brightness_range=[0.98, 1.5],
    horizontal_flip=False,
    vertical_flip=False,
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.25)

def getMeanStd():
    with open('meanClassification.pickle', 'rb') as file:
        mean = pickle.load(file)

    with open('stdClassification.pickle', 'rb') as file:
        std = pickle.load(file)

    return mean, std


mean,std=getMeanStd()
datagen.mean=mean
datagen.std=std

batchSize=32
numClasses = 5
width=240
height=240
input_shape=(width,height,3)

train_generator = datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='../data/resized_train_cropped',
        x_col="image",
        y_col="level",
        target_size=(width,height),
        batch_size=16,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        subset='training')

validation_generator =datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='../data/resized_train_cropped',
        x_col="image",
        y_col="level",
        target_size=(width,height),
        batch_size=16,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        subset='validation')
#----------------------------------------------------------------------------------------

def build_model(hp):
    weight_decay = 1e-4
    L2_norm = regularizers.l2(weight_decay)

    input = Input(shape=input_shape)
    print(input)

    x = Conv2D(96, (9, 9), activation='relu', name='conv1', kernel_regularizer=L2_norm)(input)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    x = Conv2D(384, (5, 5), activation='relu', name='conv2', kernel_regularizer=L2_norm)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    x = Conv2D(384, (3, 3), activation='relu', name='conv3')(x)
    x = Conv2D(384, (3, 3), activation='relu', name='conv4')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv5')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool3')(x)

    x = Flatten()(x)
    #x = Dense(4096, activation='relu', name='fc1')(x)
    #header--------------------------------------------------------------------------------


    for i in range(hp.Int('n_layers',1,3)):
        outLayers = Dense(hp.Int(f'conv_{i}_units',min_value=32,max_value=1024,step=64), activation='relu', name=f'fc_{i}')(x)

    outLayers = Dropout(hp.Float(f'dropout{1}', min_value=0.1, max_value=0.3, step=0.1))(outLayers)
    outLayers = Dense(numClasses, activation='softmax', name='predictions')(outLayers)

    model=Model(input,outLayers)

    optimizer = k.optimizers.SGD(learning_rate=hp.Choice('learning_rate',values=[1e-4, 1e-5,1e-6]))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model

hp=HyperParameters()


tuner = Hyperband(
    build_model,
    objective='acc',
    max_epochs=2,
    executions_per_trial=1,
    directory=os.path.normpath('F:/TFGMODELS/tuner'),
    project_name='tfg3'
)

tuner.search(train_generator,
             epochs=2,
             validation_data=validation_generator
             )

tuner.search_space_summary()

models = tuner.get_best_models(num_models=1)[0]

tuner.results_summary()