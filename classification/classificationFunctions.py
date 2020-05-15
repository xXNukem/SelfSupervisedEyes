from os import listdir,path
import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import img_to_array
import csv
import pickle
from PIL import Image

class classification:

    def calculateMeanStdClassification(self,path):
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
                print('Procesando imagen: ', img_name)
                train_mean[0, 0, channel] += np.mean(img[:, :, channel])
                train_std[0, 0, channel] += np.std(img[:, :, channel])
                print('Media acumulada: ', train_mean)
                print('Desviacion típica acumulada:', train_std)

        train_mean = train_mean / n
        train_std = train_std / n

        with open('../classification/meanClassification.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../classification/stdClassification.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        print('MEAN RGB saved meanClassification.pickle :')
        print(train_mean)
        print('STD RGB saved stdClassification.pickle:')
        print(train_std)

    def getContextPredictionMeanStd(self):

        with open('contextPredictionMean.pickle', 'rb') as file:
            mean = pickle.load(file)

        with open('contextPredictionStd.pickle', 'rb') as file:
            std = pickle.load(file)

        return mean, std

    def getJiggsawMeanStd(self):

        with open('jiggsawMean.pickle', 'rb') as file:
            mean = pickle.load(file)

        with open('jiggsawStd.pickle', 'rb') as file:
            std = pickle.load(file)

        return mean, std

    def getRotationMeanStd(self):

        with open('rotationMean.pickle', 'rb') as file:
            mean = pickle.load(file)

        with open('rotationStd.pickle', 'rb') as file:
            std = pickle.load(file)

        return mean, std