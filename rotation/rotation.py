from __future__ import absolute_import
from __future__ import print_function
import importlib.util
spec = importlib.util.spec_from_file_location("rotationNetwork.py", "../neuralNetworks/rotationNetwork.py")
rotationNetwork = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotationNetwork)
import dataGenerator
import tensorflow.keras as k
from keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import rotationFunctions
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

numClasses = 3
batchSize = 16
width=240
height=240
input_shape=(width,height,3)
network=rotationNetwork.rotationNetwork(input_shape,numClasses)
rotation=rotationFunctions.rotation()
model=network.getNetwork()

train,validation=rotation.getTrainValidationSplits()

#creacion de ID List
ID_List_train=[]
ID_List_val=[]
for i in range(0,len(train)):
    ID_List_train.append(int(i))
for i in range(0,len(validation)):
    ID_List_val.append((int(i)))


datagen = ImageDataGenerator(#rescale=1./255.0,
                            zoom_range=[-5.0, 5.0],
                             width_shift_range=[-40, 40],
                             height_shift_range=[-40, 40],
                             rotation_range=50,
                             shear_range=50,
                             horizontal_flip=False,
                             vertical_flip=False,
                             #brightness_range=[0.98,1.05],
                             # channel_shift_range=1.5
                             )


params = {'dim': (width,height),
          'batch_size':batchSize,
          'n_classes': numClasses,
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

parada=callbacks.callbacks.EarlyStopping(monitor='val_acc',mode='max',verbose=1,restore_best_weights=False,patience=3)
learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='val_acc', verbose=1, mode='max',factor=0.2, min_lr=1e-11,patience=3)


model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=1000,
                        epochs=40,
                        validation_steps=1000,
                        callbacks=[learningRate]
                         )

#train samples= 26332
#validation samples=8776                                #validation_samples
classes,samples=validation_generator.getClasses()
Y_pred = model.predict_generator(validation_generator, samples // batchSize+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
pred=[]
for i in range(0,samples):
    pred.append(y_pred[i])
print(confusion_matrix(classes, pred))
print('Classification Report')
target_names = ['0', '1', '2']
print(classification_report(classes, pred, target_names=target_names))


model.save('./rotationModel4.h5')
model.save_weights('./rotationModelWeights4.h5')
