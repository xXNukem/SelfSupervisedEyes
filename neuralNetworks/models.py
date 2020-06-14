from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D,MaxPooling2D, Concatenate, BatchNormalization
from keras.applications import vgg16

class models:

    def __init__(self,inputShape,numClasses):
        self.input_shape=inputShape
        self.numClasses=numClasses


    # VGG16 convolutional layers
    def createBaseNetwork(self):
        input = Input(shape=self.input_shape)
        print(input)
        basemodel = Sequential()
        baseModel = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input)

        return baseModel

    #add layers to VGG16 for rotation
    def getRotationNetwork(self):
        base_network = self.createBaseNetwork()

        input_a = Input(shape=self.input_shape)

        input = base_network(input_a)
        outlayers=Flatten()(input)
        outLayers = Dense(1024, activation='relu', name='fc1')(outlayers)
        outLayers = Dropout(0.2)(outLayers)
        outLayers = Dense(512, activation='relu', name='fc2')(outLayers)
        outLayers = Dense(self.numClasses, activation='softmax', name='predictions')(outLayers)

        model = Model(input_a, outLayers)

        return model

    #get siamese network for context prediction
    def getSiameseCPNetwork(self):
        base_network = self.createBaseNetwork()

        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        left = base_network(input_a)
        right = base_network(input_b)

        outLayers = Concatenate(axis=-1)([left, right])
        outLayers = Flatten() (outLayers)
        outLayers = Dense(2048, activation='relu', name='fc3')(outLayers)
        outLayers = Dropout(0.2)(outLayers)
        outLayers = Dense(1024, activation='relu', name='fc4')(outLayers)
        outLayers = Dropout(0.2)(outLayers)
        outLayers = Dense(512, activation='relu', name='fc5')(outLayers)
        outLayers = Dense(self.numClasses, activation='softmax', name='predictions')(outLayers)

        model = Model([input_a, input_b], outLayers)

        return model

    #get siamese network for jiggsaw
    def getSiameseJiggsawNetwork(self):
        base_network = self.createBaseNetwork()

        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        input_c = Input(shape=self.input_shape)
        input_d = Input(shape=self.input_shape)

        a = base_network(input_a)
        b = base_network(input_b)
        c = base_network(input_c)
        d = base_network(input_d)

        outLayers = Concatenate(axis=-1)([a,b,c,d])
        outLayers = Flatten()(outLayers)
        outLayers = Dense(1024, activation='relu', name='fc4')(outLayers)
        outLayers = Dropout(0.2)(outLayers)
        outLayers = Dense(512, activation='relu', name='fc5')(outLayers)
        outLayers = Dense(self.numClasses, activation='softmax', name='predictions')(outLayers)

        model = Model([input_a, input_b,input_c,input_d], outLayers)

        return model