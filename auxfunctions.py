from os import listdir,path
import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import img_to_array
import csv
import pickle

def splitfilename(filename):
    sname=""
    sext=""
    i=filename.rfind(".")
    if(i!=0):
        n=len(filename)
        j=n-i-1
        sname=filename[0:i]
        sext=filename[-j:]
    return sname, sext

#Carga una lista de tuplas con las rutas a las imágenes emparejadas
def loadimgspath(path):

    X=[]

    for i in listdir(path):
        imgs=listdir(path+'/'+i)
        for j in ['0','1','2','3','4','5','6','7']:
                #Centro                 cuadrado                    etiqueta
            aux=(path+'/'+i+'/'+imgs[8],path+'/'+i+'/'+imgs[int(j)],j)
            print('Pair created: ',aux)
            X.append(aux)

    return X

#Divide el conjunto de datos en entrenamiento y validacion
def splitGenerator(imglist,percent):

    print('Splitting the dataset in train/validation')
    train = []
    validation = []
    validationPercent = (percent * int(len(imglist))) / 100
    for i in range(1, int(validationPercent)):
        index = random.randrange(int(len(imglist)))
        aux = imglist[index]
        if aux not in validation:
            validation.append(aux)
            imglist.pop(index)
        else:
            i = i - 1
    train = imglist

    with open('train.pickle', 'wb') as file:
        pickle.dump(train, file, pickle.HIGHEST_PROTOCOL)

        print('train.pickle saved')

    with open('validation.pickle', 'wb') as file:
        pickle.dump(validation, file, pickle.HIGHEST_PROTOCOL)

        print('validation.pickle saved')

    #return train, validation

#Lee los csv con los splits de train y validacion y los devuelve como tuplas para pasar al DataGenerator
def getTrainValidationSplits():
    with open('train.pickle', 'rb') as file:
        train = pickle.load(file)

    with open('validation.pickle', 'rb') as file:
        validation = pickle.load(file)

    return train,validation


#Obtenemos media y desviacion tipica de las imagenes
def getMeanStd(path):

    train_mean = np.zeros((1,1,3))
    train_std = np.zeros((1,1,3))

    n = 0
    for folder in os.listdir('./'+path):
        for img in os.listdir('./'+path+'/'+folder):
            n=n+1
            image = cv2.imread('./'+path+'/'+folder+'/'+img)
            toarray = img_to_array(image)
            for channel in range(toarray.shape[2]):
                print('Processing image: ',img)
                train_mean[0,0,channel] += np.mean(toarray[:,:,channel])
                train_std[0,0,channel] += np.std(toarray[:,:,channel])
                print('Acc Mean: ',train_mean)
                print('Acc STD:',train_std)

    train_mean = train_mean / n
    train_std = train_std / n

    print('MEAN RGB:')
    print(train_mean)
    print('STD RGB:')
    print(train_std)

    return train_mean, train_std
