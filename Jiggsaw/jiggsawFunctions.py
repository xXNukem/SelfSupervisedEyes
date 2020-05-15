from os import listdir
import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import img_to_array
import auxfunctions
import pickle
import numpy as np
from itertools import permutations
from PIL import Image

class jiggsaw:

    def generateDataset(self,imgPath, sqSize, pathname):

        if os.path.exists(pathname) == False:
            os.mkdir(pathname)

        imgDir = listdir(imgPath)
        # tamaño del cuadrado en pixels
        squareSize = sqSize
        # Porcentaje de pixels que se pueden desplazar como maximo

        for i in imgDir:
            name, exte = auxfunctions.splitfilename(i)  # extrayendo el nombre del archivo sin la extensión
            dir =pathname + '/' + name
            os.mkdir(dir)

            print('Generating crops for:', i)
            img = Image.open(imgPath + '/' + i)
            assert 3 * squareSize <= img.width * img.height
            # --------------------------
            #Pixeles desde el punto central
            randX = random.uniform(1, int(sqSize/5))
            randY = random.uniform(1, int(sqSize/5))

            randomOrientation = random.randint(0,
                                               8)  # Genero aleatoriamente hacia donde quiero que vaya más o menos el recorte central
            print('Orientation:', randomOrientation)

            # El recorte central se desplaza hacia la izquierda
            if randomOrientation == 1:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se desplaza hacia la diagonal superior izquierda
            if randomOrientation == 2:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se deplaza hacia arriba
            if randomOrientation == 3:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se desplaza a la diagonal superior derecha
            if randomOrientation == 4:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se desplaza a la derecha
            if randomOrientation == 5:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se desplaza a la exquina inferior derecha
            if randomOrientation == 6:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se desplaza hacia abajo
            if randomOrientation == 7:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se desplaza a la diagonal inferior izquierda
            if randomOrientation == 8:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)

            # El recorte central se queda centrado
            if randomOrientation == 0:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)


            # Obtencion del recorte 0 (arriba izquierda)

            newXCenter = xCenter - int(sqSize/2)
            newYCenter = yCenter - int(sqSize/2)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            upLeftSquare = img.crop(croppingMask)

            upLeftSquare.save(pathname + '/' + name + '/0.jpg')

            # Obtencion del recorte 1 (arriba derecha)

            newXCenter = xCenter + int(sqSize/2)
            newYCenter = yCenter - int(sqSize/2)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            upRightSquare = img.crop(croppingMask)

            upRightSquare.save(pathname + '/' + name + '/1.jpg')

            # Obtencion del recorte 2 (abajo derecha)

            newXCenter = xCenter + int(sqSize/2)
            newYCenter = yCenter + int(sqSize/2)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            downRightSquare = img.crop(croppingMask)

            downRightSquare.save(pathname + '/' + name + '/2.jpg')

            # Obtencion del recorte 3 (abajo izquierda)

            newXCenter = xCenter - int(sqSize/2)
            newYCenter = yCenter + int(sqSize/2)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            downRightSquare = img.crop(croppingMask)

            downRightSquare.save(pathname + '/' + name + '/3.jpg')

    def mj_dist_perms(self,p1, p2):
        d = 0
        for i in range(len(p1)):
            if p1[i] != p2[i]:
                d += 1
        return d

    def getPermutationList(self,maxDist):
        ncells = 4

        x = list(range(ncells))

        perms = list(permutations(x, ncells))

        nperms = len(perms)

        # Compute distances
        D = np.zeros((nperms, nperms)) - 1

        for i in range(0, nperms - 1):
            for j in range(i + 1, nperms):
                D[i, j] = self.mj_dist_perms(perms[i], perms[j])

        # Select the top K
        l_sel_perms = [perms[0]]
        K = maxDist

        cdist = ncells

        while len(l_sel_perms) < K:
            # Could be done more efficiently
            for i in range(0, nperms - 1):
                for j in range(i + 1, nperms):
                    if D[i, j] == cdist:
                        l_sel_perms.append(perms[j])
                        if len(l_sel_perms) >= K:
                            break
                if len(l_sel_perms) >= K:
                    break
            cdist -= 1

            if cdist < 2:
                break

        print("Permutation List obtained!")
        return l_sel_perms


    # Carga una lista de tuplas con las rutas a las imágenes emparejadas
    def loadimgspath(self, path,maxDist):
        permutationList=self.getPermutationList(maxDist)
        X = []

        for i in listdir(path):
            imgs = listdir(path + '/' + i)
            for j in imgs:
               randomPerm=random.randint(0,len(permutationList)-1) #randomPerm será a su vez la etiqueta
               perm=permutationList[randomPerm] #guardo la permutacion correspondiente a la etiqueta
               aux=(path + '/' + i+'/'+str(perm[0])+'.jpg',path + '/' + i+'/'+str(perm[1])+'.jpg',path + '/' + i+'/'+str(perm[2])+'.jpg',path + '/' + i+'/'+str(perm[3])+'.jpg',str(randomPerm))
               print('Perm created: ',aux)
               X.append(aux)

        return X

        # Divide el conjunto de datos en entrenamiento y validacion
    def splitGenerator(self, imglist, percent):

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

        with open('../jiggsaw/train.pickle', 'wb') as file:
            pickle.dump(train, file, pickle.HIGHEST_PROTOCOL)

            print('train.pickle saved')

        with open('../jiggsaw/validation.pickle', 'wb') as file:
            pickle.dump(validation, file, pickle.HIGHEST_PROTOCOL)

            print('validation.pickle saved')

        # Lee los csv con los splits de train y validacion y los devuelve como tuplas para pasar al DataGenerator
    def getTrainValidationSplits(self):
        with open('train.pickle', 'rb') as file:
            train = pickle.load(file)

        with open('validation.pickle', 'rb') as file:
            validation = pickle.load(file)

        return train, validation

    # Obtenemos media y desviacion tipica de las imagenes
    def calculateMeanStd(self, path):

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

        with open('../jiggsaw/jiggsawMean.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../jiggsaw/jiggsawStd.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        with open('../classification/jiggsawMean.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../classification/jiggsawStd.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        print('MEAN RGB saved mean.pickle :')
        print(train_mean)
        print('STD RGB saved std.pickle:')
        print(train_std)

    def getMeanStd(self):

        with open('jiggsawMean.pickle', 'rb') as file:
            mean = pickle.load(file)

        with open('jiggsawStd.pickle', 'rb') as file:
             std = pickle.load(file)

        return mean, std