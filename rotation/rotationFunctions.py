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

        tools = imgTools.imgTools()
        if os.path.exists(pathname) == False:
            os.mkdir(pathname)
            os.mkdir(pathname + '/' + '0')
            os.mkdir(pathname + '/' + '1')
            os.mkdir(pathname + '/' + '2')
        imgs = listdir(imgPath)

        for img in imgs:
            name, ext = auxfunctions.splitfilename(img)
            completePath = imgPath + '/' + img
            print('Rotating: ', completePath)
            angle = choice([0, 90, -90])
            rotated = tools.rotate(completePath, angle)
            if angle == 0:
                cv2.imwrite(pathname + '/0/' + name + '_' + '0' + '.jpeg', rotated)
            elif angle == 90:
                cv2.imwrite(pathname + '/1/' + name + '_' + '1' + '.jpeg', rotated)
            elif angle == -90:
                cv2.imwrite(pathname + '/2/' + name + '_' + '2' + '.jpeg', rotated)

        # Obtenemos media y desviacion tipica de las imagenes

    def calculateMeanStd(self, path):
        # Get mean, std of R,G,B images
        train_mean = np.zeros((1, 1, 3))
        train_std = np.zeros((1, 1, 3))

        n = 0
        for folder in os.listdir(path):
            for img in os.listdir(path + '/' + folder):
                n = n + 1
                image = cv2.imread(path + '/' + folder + '/' + img)
                toarray = img_to_array(image)
                for channel in range(toarray.shape[2]):
                    print('Processing image: ', img)
                    train_mean[0, 0, channel] += np.mean(toarray[:, :, channel])
                    train_std[0, 0, channel] += np.std(toarray[:, :, channel])
                    print('Acc Mean: ', train_mean)
                    print('Acc STD:', train_std)

        train_mean = train_mean / n
        train_std = train_std / n

        with open('../rotation/rotationMean.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../rotation/rotationStd.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        with open('../classification/rotationMean.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../classification/rotationStd.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        print('MEAN RGB saved meanClassification.pickle :')
        print(train_mean)
        print('STD RGB saved stdClassification.pickle:')
        print(train_std)

    def getMeanStd(self):

        with open('rotationMean.pickle', 'rb') as file:
            mean = pickle.load(file)

        with open('rotationStd.pickle', 'rb') as file:
            std = pickle.load(file)

        return mean, std