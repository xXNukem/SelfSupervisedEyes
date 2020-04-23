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

numClasses = 8
dataset='./retinopatyDataset'
width=96 #diabetic retinopaty 120 120, drRafael 40 40, 96 96
height=96
input_shape=(width,height,3)

#Funcion que define la red siamesa
def createBaseNetwork(input_shape):
    weight_decay = 1e-4
    L2_norm = regularizers.l2(weight_decay)

    input = Input(shape=input_shape)
    print(input)
    #x = Flatten()(input)
    x = Conv2D(96, (9,9), activation='relu',name='conv1',kernel_regularizer=L2_norm)(input)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    #LRN1 (investigar)
    x = Conv2D(384, (5,5), activation='relu',name='conv2',kernel_regularizer=L2_norm)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    #LRN2 (investigar)
    x = Conv2D(384, (3, 3), activation='relu',name='conv3')(x)
    x = Conv2D(384, (3, 3), activation='relu', name='conv4')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv5')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool3')(x)
    #x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Flatten()(x)
    x= Dense(4096,activation='relu',name='fc1')(x)

    return Model(input, x)

#---------------------------------------------------------------------------------
def getSiameseNetWork(input_shape,numClasses):
    base_network = createBaseNetwork(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    #Reuso de la instancia para compartir los pesos
    left = base_network(input_a)
    right = base_network(input_b)

    #Concatenar con la red siamesa y generar capas adicionales
    outLayers=Concatenate(axis=-1)([left,right])
    outLayers=Dense(4096, activation='relu', name='fc2')(outLayers)
    outLayers=Dropout(0.2)(outLayers)
    outLayers=Dense(2048, activation='relu', name='fc3')(outLayers)
    outLayers = Dropout(0.2)(outLayers)
    outLayers=Dense(1024, activation='relu', name='fc4')(outLayers)
    outLayers = Dropout(0.2)(outLayers)
    outLayers = Dense(512, activation='relu', name='fc5')(outLayers)
    outLayers=Dense(numClasses, activation='softmax', name='predictions')(outLayers)

    model = Model([input_a, input_b], outLayers)

    return model

#----------------------------------------------------------------------------------------
    
model=getSiameseNetWork(input_shape,numClasses)

#Obtencion de la lista de tuplas con las rutas de las imagenes
#imgList=auxfunctions.loadimgspath(dataset)
#Obtencion del conjutno de entrenamiento y validacion con un 25% en validacion
train,validation=auxfunctions.getTrainValidationSplits()

#creacion de ID List
ID_List_train=[]
ID_List_val=[]
for i in range(0,len(train)):
    ID_List_train.append(int(i))
for i in range(0,len(validation)):
    ID_List_val.append((int(i)))


params = {'dim': (width,height),
          'batch_size':64,
          'n_classes': 8,
          'n_channels': 3,
          'shuffle': True,
          'normalize': True,
          'downsampling':True,
          'downsamplingPercent':65}

training_generator=dataGenerator.DataGenerator(train,ID_List_train,**params)
validation_generator=dataGenerator.DataGenerator(validation,ID_List_val,**params)

optimizer=k.optimizers.Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc'])


model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=1000,
                    epochs=25)

model.save('./Model2.h5')
model.save_weights('./ModelWeights2.h5')