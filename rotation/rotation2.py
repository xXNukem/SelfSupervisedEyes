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
batchSize = 32
width=240
height=240
input_shape=(width,height,3)
network=rotationNetwork.rotationNetwork(input_shape,numClasses)
rotation=rotationFunctions.rotation()
model=network.getNetwork()

mean,std=rotation.getMeanStd()
train_DataGen = ImageDataGenerator(
    rescale=1. / 255,
    height_shift_range=[-40, 40],
    width_shift_range=[-40, 40],
    shear_range=45,
    zoom_range=[-3.5,3.5],
    rotation_range=50,
    horizontal_flip=False,
    vertical_flip=False,
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.25)

train_DataGen.mean=mean
train_DataGen.std=std

"""Generamos las imagenes con los parametros establecidos"""
training_generator = train_DataGen.flow_from_directory(
    '../data/240x240resized_train_cropped_rotated',
    target_size=(height, width),
    batch_size=batchSize,
    class_mode='categorical',
    color_mode='rgb',
    subset='training')

validation_generator = train_DataGen.flow_from_directory(
    '../data/240x240resized_train_cropped_rotated',
    target_size=(height, width),
    batch_size=batchSize,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation')


optimizer=k.optimizers.Adam(learning_rate=1e-5)
model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['sparse_categorical_accuracy']) #probar

parada=callbacks.callbacks.EarlyStopping(monitor='val_acc',mode='max',verbose=1,restore_best_weights=False,patience=2)
learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='val_acc', verbose=1, mode='max',factor=0.2, min_lr=1e-11,patience=2)


model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=1000,
                        epochs=20,
                        validation_steps=1000,
                        shuffle=True,
                        callbacks=[learningRate,parada]
                         )

#train samples= 26332
#validation samples=8776                                #validation_samples
Y_pred = model.predict_generator(validation_generator, 8776 // batchSize+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['0º', '90º', '-90º']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


model.save('./rotationModel4.h5')
model.save_weights('./rotationModelWeights4.h5')
