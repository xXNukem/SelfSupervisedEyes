from __future__ import absolute_import
from __future__ import print_function
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D,MaxPooling2D, Concatenate, BatchNormalization
from tensorflow.keras import regularizers

class contextPredictionNetwork:

    def __init__(self,inputShape,numClasses):
        self.input_shape=inputShape
        self.numClasses=numClasses


    # Funcion que define la red siamesa
    def createBaseNetwork(self):
        weight_decay = 1e-4
        L2_norm = regularizers.l2(weight_decay)

        input = Input(shape=self.input_shape)
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
        x = Dense(4096, activation='relu', name='fc1')(x)

        return Model(input, x)

    # ---------------------------------------------------------------------------------
    def getSiameseNetwork(self):
        base_network = self.createBaseNetwork()

        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        # Reuso de la instancia para compartir los pesos
        left = base_network(input_a)
        right = base_network(input_b)

        # Concatenar con la red siamesa y generar capas adicionales
        outLayers = Concatenate(axis=-1)([left, right])
        outLayers = Dense(4096, activation='relu', name='fc2')(outLayers)
        outLayers = Dropout(0.2)(outLayers)
        outLayers = Dense(2048, activation='relu', name='fc3')(outLayers)
        outLayers = Dropout(0.2)(outLayers)
        outLayers = Dense(1024, activation='relu', name='fc4')(outLayers)
        outLayers = Dropout(0.2)(outLayers)
        outLayers = Dense(512, activation='relu', name='fc5')(outLayers)
        outLayers = Dense(self.numClasses, activation='softmax', name='predictions')(outLayers)

        model = Model([input_a, input_b], outLayers)

        return model

    # ----------------------------------------------------------------------------------------