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
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas
from tensorflow.keras.callbacks import TensorBoard
import datetime
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
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
    validation_split=0.10)

mean,std=auxfunctions.getMeanStdClassification()
datagen.mean=mean
datagen.std=std

numClasses = 5
width=240 #diabetic retinopaty 120 120, drRafael 40 40, 96 96
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
        color_mode='rgb', #quitar o no quitar
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
        subset='validation')
#----------------------------------------------------------------------------------------

def build_model(hp):

    input=Input(input_shape)
    model = Conv2D(96, (9, 9), activation='relu', padding='same', name='conv1')(input)
    model=Dense(numClasses, activation='softmax', name='predictions')(model)

    model=Model(input,model)

    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1,
    directory='./logtunner'
)
