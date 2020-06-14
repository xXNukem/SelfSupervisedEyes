from __future__ import absolute_import
from __future__ import print_function
import importlib.util
spec = importlib.util.spec_from_file_location("models.py", "../neuralNetworks/models.py")
rotationNetwork = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotationNetwork)
import keras as k
from keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import TensorBoard
import datetime
import click

"""
PARAMS:
img_size: Size of the imput img, recomended 120x120, 96x96 or 240x240
epochs: Number of epochs
steps: Number of steps per epoch
batch_size: Number of images per batch, recomeded less than 64
eta: Learning Rate value
model_name: Name for the generated files (_HIST,_WEIGHTS,_MODEL)

In this training you have to enter the path for the TRAIN FILES and VALIDATION FILES 
with the train_files and validation_files argument
"""

#params
@click.command()
@click.option('--train_files', '-T', default=None, required=True,help=u'Training files.')
@click.option('--validation_files', '-V', default=None, required=True, help=u'Validation files.')
@click.option('--img_size', '-S', default=None, required=True, help=u'SizeXSize.')
@click.option('--epochs', '-E', default=400, required=False, help=u'Number of epochs')
@click.option('--steps', '-s', default=250, required=False, help=u'Steps per epoch')
@click.option('--eta', '-e', default=(1e-5), required=False, help=u'Eta value for learning rate')
@click.option('--batch_size', '-B', default=64, required=False, help=u'Batch Size')
@click.option('--model_name', '-N', default='model', required=False,help=u'Name for the saved model and hist.')
@click.option('--val_split', '-v', default=0.25, required=False, help=u'Batch Size')

def launchTraining(train_files,validation_files,img_size,
                   epochs,steps,eta,batch_size,model_name,val_split):
    numClasses = 3
    batchSize = int(batch_size)
    width=int(img_size)
    height=int(img_size)
    input_shape=(width,height,3)
    network=rotationNetwork.models(input_shape,numClasses)
    model=network.getRotationNetwork()

    train_DataGen = ImageDataGenerator(
        rescale=1./255.0,
        zoom_range=0.2,
        height_shift_range=[-1.5, 1.5],
        width_shift_range=[-1.5, 1.5],
        shear_range=2,
        #rotation_range=2,
        channel_shift_range=1,
        brightness_range=[0.98,1.5],
        horizontal_flip=False,
        vertical_flip=False,
        validation_split=float(val_split))

    #With this data generation, all images will be resized
    training_generator = train_DataGen.flow_from_directory(
        train_files,
        target_size=(height, width),
        batch_size=batchSize,
        class_mode='categorical',
        color_mode='rgb',
        subset='training')

    validation_generator = train_DataGen.flow_from_directory(
        validation_files,
        target_size=(height, width),
        batch_size=batchSize,
        class_mode='categorical',
        color_mode='rgb',
        subset='validation')


    optimizer=k.optimizers.Adam(learning_rate=eta)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])

    log_dir="logs\\fit\\" +'rotation_lr4'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    parada=callbacks.callbacks.EarlyStopping(monitor='val_loss',
                                             mode='min',verbose=1,
                                             restore_best_weights=True,
                                             patience=10)
    learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       verbose=1,
                                                       mode='min',
                                                       factor=0.2,
                                                       min_lr=1e-8,
                                                       patience=5)


    history=model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=int(steps),
                            epochs=int(epochs),
                            validation_steps=int(steps),
                            verbose=1,
                            shuffle=True,
                            callbacks=[learningRate,parada,tensorboard_callback]
                             )
    #save model results
    np.save(model_name + '_HIST.npy', history.history)
    model.save(model_name + '_MODEL.h5')
    model.save_weights(model_name + '_WEIGHTS.h5')

launchTraining()