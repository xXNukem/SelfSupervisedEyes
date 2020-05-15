from __future__ import absolute_import
from __future__ import print_function
import importlib.util
spec = importlib.util.spec_from_file_location("contexPredictionFunctions.py", "../neuralNetworks/contextPredictionNetwork.py")
contextPredictionNetwork = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contextPredictionNetwork)
import dataGenerator
import tensorflow.keras as k
from keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import contextPredictionFunctions

numClasses = 8
dataset='../retinopatyDataset'
width=96 #diabetic retinopaty 120 120, drRafael 40 40, 96 96
height=96
input_shape=(width,height,3)
contextPrediction=contextPredictionFunctions.contextPrediction()
network=contextPredictionNetwork.contextPredictionNetwork(input_shape,numClasses)

model=network.getSiameseNetwork()

train,validation=contextPrediction.getTrainValidationSplits()

#creacion de ID List
ID_List_train=[]
ID_List_val=[]
for i in range(0,len(train)):
    ID_List_train.append(int(i))
for i in range(0,len(validation)):
    ID_List_val.append((int(i)))


datagen = ImageDataGenerator(
                            zoom_range=[-1.5, 1.5],
                             width_shift_range=[-1.5, 1.5],
                             height_shift_range=[-1.5, 1.5],
                             rotation_range=2,
                             shear_range=2,
                             horizontal_flip=False,
                             vertical_flip=False,
                             #brightness_range=[0.98,1.05],
                             # channel_shift_range=1.5
                             )


params = {'dim': (width,height),
          'batch_size':64,
          'n_classes': 8,
          'n_channels': 3,
          'shuffle': True,
          'normalize': True,
          'downsampling':True,
          'downsamplingPercent':80,
          'dataAugmentation':True,
            'rgbToGray':True ,
          'datagen':datagen
            }


training_generator=dataGenerator.DataGenerator(train,ID_List_train,**params)
validation_generator=dataGenerator.DataGenerator(validation,ID_List_val,**params)


optimizer=k.optimizers.Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc'])

parada=callbacks.callbacks.EarlyStopping(monitor='val_acc',mode='max',verbose=1,restore_best_weights=True,patience=3)
learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='val_acc', verbose=1, mode='max',factor=0.2, min_lr=1e-8,patience=3)


model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=1000,
                        epochs=40,
                        validation_steps=1000,
                        callbacks=[parada,learningRate]
                         )

model.save('./contextPredictionModel.h5')
model.save_weights('./contextPredictionModelWeights.h5')