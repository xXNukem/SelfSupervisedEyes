import importlib.util
spec = importlib.util.spec_from_file_location("imgTools.py", "../main/imgTools.py")
imgTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imgTools)
from os import listdir
import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import img_to_array
import auxfunctions
import pickle
from PIL import Image
from random import choice

class rotation:

    def generateDataset(self,imgPath,pathname):

        tools=imgTools.imgTools()
        if os.path.exists(pathname) == False:
            os.mkdir(pathname)

        imgs=listdir(imgPath)

        for img in imgs:
            name,ext=auxfunctions.splitfilename(img)
            completePath=imgPath+'/'+img
            print('Rotating: ',completePath)
            angle=choice([0,90,180,270])
            rotated=tools.rotate(completePath,angle)
            cv2.imwrite(pathname+'/'+name+'_'+str(angle)+'.jpeg', rotated)

    # Carga una lista de tuplas con las rutas a las imágenes emparejadas
    def loadimgspath(self,path):

        X = []

        for i in listdir(path):
            completePath=path+'/'+i
            angle=auxfunctions.splitGetAngle(i)
            aux=(completePath,angle)
            print('Pair created: ',aux)
            X.append(aux)

        return X

    # Divide el conjunto de datos en entrenamiento y validacion
    def splitGenerator(self,imglist, percent):

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

        with open('../rotation/train.pickle', 'wb') as file:
            pickle.dump(train, file, pickle.HIGHEST_PROTOCOL)

            print('train.pickle saved')

        with open('../rotation/validation.pickle', 'wb') as file:
            pickle.dump(validation, file, pickle.HIGHEST_PROTOCOL)

            print('validation.pickle saved')

    def getTrainValidationSplits(self):
        with open('train.pickle', 'rb') as file:
            train = pickle.load(file)

        with open('validation.pickle', 'rb') as file:
            validation = pickle.load(file)

        return train, validation

        # Obtenemos media y desviacion tipica de las imagenes

    def calculateMeanStd(self, path):
        # Get mean, std of R,G,B images
        train_mean = np.zeros((1, 1, 3))
        train_std = np.zeros((1, 1, 3))

        imgs = os.listdir(path)
        n = len(imgs)
        for img_name in imgs:
            img = cv2.imread(path + '/' + img_name)
            img = img_to_array(img)
            # img /= 255.
            for channel in range(img.shape[2]):
                print('Processing: ', img_name)
                train_mean[0, 0, channel] += np.mean(img[:, :, channel])
                train_std[0, 0, channel] += np.std(img[:, :, channel])
                print('Mean acc: ', train_mean)
                print('STD acc:', train_std)

        train_mean = train_mean / n
        train_std = train_std / n

        with open('../rotation/mean.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../rotation/std.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        print('MEAN RGB saved meanClassification.pickle :')
        print(train_mean)
        print('STD RGB saved stdClassification.pickle:')
        print(train_std)

    def getMeanStd(self):

        with open('mean.pickle', 'rb') as file:
            mean = pickle.load(file)

        with open('std.pickle', 'rb') as file:
            std = pickle.load(file)

        return mean, std