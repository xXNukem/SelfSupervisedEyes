import click
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
import datetime
from keras.callbacks import TensorBoard
from keras import callbacks
import tensorflow.keras as k
import numpy as np

#params
@click.command()
@click.option('--train_files', '-T', default=None, required=True,help=u'Training files.')
@click.option('--validation_files', '-V', default=None, required=True, help=u'Validation files.')
@click.option('--classification', '-c', is_flag=True,help=u'-c for classification training, nothing for regression training.')
@click.option('--imagenet', '-i', is_flag=True,help=u'-i to load imagenet weights, nothing to load your own weights or start with random weights')
@click.option('--load_weights', '-l', is_flag=True,help=u'-l for enable weight loading.')
@click.option('--weights', '-w', default=None, required=False, help=u'file with the weights to load.')
@click.option('--eta', '-e', default=(1e-5), required=False, help=u'Eta value for learning rate')
@click.option('--model_name', '-N', default='model', required=False,help=u'Name for the saved model and hist.')
@click.option('--img_size', '-S', default=None, required=True, help=u'SizeXSize.')
@click.option('--batch_size', '-B', default=16, required=False, help=u'Batch Size')
@click.option('--epochs', '-E', default=400, required=False, help=u'Number of epochs')
@click.option('--steps', '-s', default=250, required=False, help=u'Steps per epoch')
@click.option('--val_split', '-v', default=0.25, required=False, help=u'Batch Size')

#launch main training for regresion or classification
def launchTraining(train_files,validation_files,classification,
                   eta,load_weights,weights,model_name,imagenet,
                   img_size,batch_size,epochs,steps,val_split):

    train_datagen = ImageDataGenerator(
        zoom_range=0.2,
        width_shift_range=[-2, 2],
        height_shift_range=[-2, 2],
        rotation_range=3,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.90, 1.25],
        channel_shift_range=1,
        fill_mode='constant',
        cval=0,
        validation_split=float(val_split)
        )
    if classification:
        mode='categorical' #classification
    else:
        mode='sparse' #regression

    train_generator = train_datagen.flow_from_directory(
            train_files,
            target_size=(int(img_size), int(img_size)),
            batch_size=int(batch_size),
            shuffle=True,
            class_mode=mode,
            subset='training')

    validation_generator = train_datagen.flow_from_directory(
            validation_files,
            target_size=(int(img_size), int(img_size)),
            batch_size=int(batch_size),
            class_mode=mode,
            shuffle=True,
            subset='validation')

    basemodel=Sequential()
    if imagenet==False:
        baseModel=vgg16.VGG16(weights=None, include_top=False,
                              input_tensor=Input(shape=(224, 224, 3)))
    else:
        baseModel = vgg16.VGG16(weights='imagenet', include_top=False,
                                input_tensor=Input(shape=(224, 224, 3)))
    if load_weights == True:
        if weights==None:
            print('Weights file not found')
            return
        else:
            baseModel.load_weights(weights,by_name=True)

    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(1024, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(1024, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    if classification==True:
        headModel = Dense(5,activation='softmax')(headModel)
    else:
        headModel = Dense(1)(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    optimizer=k.optimizers.Adam(learning_rate=eta)
    if classification==True:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['acc'])
    else:
        model.compile(loss='huber_loss',
                    optimizer=optimizer,
                    metrics=['mean_absolute_error'])


    model.summary() #print model structure

    #callbacks
    log_dir="logs\\fit\\" +model_name+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    parada=callbacks.callbacks.EarlyStopping(monitor='val_loss',
                                             mode='min',
                                             verbose=1,
                                             restore_best_weights=True,
                                             patience=10)
    learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.1,
                                                       verbose=1,
                                                       mode='min',
                                                       min_delta=1e-10,
                                                       cooldown=0,
                                                       min_lr=0,
                                                       patience=5)


    history=model.fit_generator(
            train_generator,
            steps_per_epoch=int(steps),
            epochs=int(epochs),
            validation_data=validation_generator,
            validation_steps=int(steps),
            callbacks=[parada,learningRate,tensorboard_callback]
            )

    #save model results
    np.save(model_name+'_HIST.npy',history.history)
    model.save(model_name+'_MODEL.h5')
    model.save_weights(model_name+'_WEIGHTS.h5')

launchTraining()