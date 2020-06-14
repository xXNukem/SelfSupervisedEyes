from __future__ import absolute_import
from __future__ import print_function
import importlib.util
spec = importlib.util.spec_from_file_location("models.py", "../neuralNetworks/models.py")
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)
import dataGenerator
import keras as k
from keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import jiggsawFunctions
from keras.callbacks import TensorBoard
import datetime
import numpy as np
import click

"""
PARAMS:
img_size: Size of the imput img, recomended 120x120, 96x96 or 240x240
epochs: Number of epochs
steps: Number of steps per epoch
batch_size: Number of images per batch, recomeded less than 64
eta: Learning Rate value
model_name: Name for the generated files (_HIST,_WEIGHTS,_MODEL)

TRAIN AND VALIDATION SPLITS ARE READ FROM TRAIN.pickle file and VALIDATION.pickle file wich 
will be saved in this package after follow tutotial steps.

SIZE must be the same SQUARE SIZE or PATCH SIZE you specified when creating patches

"""

@click.command()
@click.option('--img_size', '-S', default=None, required=True, help=u'SizeXSize.')
@click.option('--epochs', '-E', default=400, required=False, help=u'Number of epochs')
@click.option('--steps', '-s', default=250, required=False, help=u'Steps per epoch')
@click.option('--eta', '-e', default=(1e-5), required=False, help=u'Eta value for learning rate')
@click.option('--batch_size', '-B', default=64, required=False, help=u'Batch Size')
@click.option('--n_classes', '-n', default=12, required=False, help=u'number of puzzle permutations')
@click.option('--model_name', '-N', default='model', required=False,help=u'Name for the saved model and hist.')

def launchTraining(img_size,epochs,steps,eta,batch_size,n_classes,model_name):
    numClasses = int(n_classes)
    width=int(img_size)
    height=int(img_size)
    input_shape=(width,height,3)
    jiggsaw=jiggsawFunctions.jiggsaw()
    network=models.models(input_shape,numClasses)

    model=network.getSiameseJiggsawNetwork()

    train,validation=jiggsaw.getTrainValidationSplits()

    #ID List Creation
    ID_List_train=[]
    ID_List_val=[]
    for i in range(0,len(train)):
        ID_List_train.append(int(i))
    for i in range(0,len(validation)):
        ID_List_val.append((int(i)))


    datagen = ImageDataGenerator(
                                zoom_range=0.2,
                                 width_shift_range=[-1.5, 1.5],
                                 height_shift_range=[-1.5, 1.5],
                                 rotation_range=2,
                                 shear_range=2,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 brightness_range=[0.98,1.05],
                                 channel_shift_range=1.5
                                 )


    params = {'dim': (width,height), #training params
              'batch_size':int(batch_size),
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


    optimizer=k.optimizers.Adam(learning_rate=eta)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])

    #callbacks
    parada=callbacks.callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,
                                             restore_best_weights=True,patience=10)
    learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, mode='min',
                                                       factor=0.2, min_lr=1e-8,patience=5)
    log_dir="logs\\fit\\"+model_name+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


    history=model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=int(steps),
                            epochs=int(epochs),
                            validation_steps=int(steps),
                            callbacks=[parada,learningRate,tensorboard_callback]
                             )

    #save training results
    np.save(model_name + '_HIST.npy', history.history)
    model.save(model_name+'_MODEL.h5')
    model.save_weights(model_name+'_WEIGHTS.h5')

launchTraining()