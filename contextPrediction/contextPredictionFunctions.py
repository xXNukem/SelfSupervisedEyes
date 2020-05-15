from os import listdir
import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import img_to_array
import auxfunctions
import pickle
from PIL import Image

class contextPrediction:

    # Carga una lista de tuplas con las rutas a las imágenes emparejadas
    def loadimgspath(self,path):

        X = []

        for i in listdir(path):
            imgs = listdir(path + '/' + i)
            for j in ['0', '1', '2', '3', '4', '5', '6', '7']:
                # Centro                 cuadrado                    etiqueta
                aux = (path + '/' + i + '/' + imgs[8], path + '/' + i + '/' + imgs[int(j)], j)
                print('Pair created: ', aux)
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

        with open('../contextPrediction/train.pickle', 'wb') as file:
            pickle.dump(train, file, pickle.HIGHEST_PROTOCOL)

            print('train.pickle saved')

        with open('../contextPrediction/validation.pickle', 'wb') as file:
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
    def calculateMeanStd(self,path):

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

        with open('../contextPrediction/ContextPredictionMean.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../contextPrediction/ContextPredictionStd.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        with open('../classification/ContextPredictionMean.pickle', 'wb') as file:
            pickle.dump(train_mean, file, pickle.HIGHEST_PROTOCOL)

        with open('../classification/ContextPredictionStd.pickle', 'wb') as file:
            pickle.dump(train_std, file, pickle.HIGHEST_PROTOCOL)

        print('MEAN RGB saved mean.pickle :')
        print(train_mean)
        print('STD RGB saved std.pickle:')
        print(train_std)

    def getMeanStd(self):

        with open('ContextPredictionMean.pickle', 'rb') as file:
            mean = pickle.load(file)


        with open('ContextPredictionStd.pickle', 'rb') as file:
            std = pickle.load(file)

        return mean, std

    def generateDataset(self, imgPath, sqSize, sqPercent, pathname):

        if os.path.exists(pathname) == False:
            os.mkdir(pathname)

        imgDir = listdir(imgPath)
        # tamaño del cuadrado en pixels
        squareSize = sqSize
        # Porcentaje de pixels que se pueden desplazar como maximo
        squareSizePercent = sqPercent
        # Pixeles que se pueden desplazar como maximo
        movementPixels = (squareSize * squareSizePercent) / 100
        assert squareSizePercent >= 1 or squareSizePercent <= 100

        """Forma de guardar los recortes de la imagen
            1   2   3
            0   C   4
            7   6   5

            -Se crea una carpeta por cada imagen
            -Se guaran los recortes con la nomenclatura de arriba
        ---------------------------------------------"""

        for i in imgDir:
            name, exte = auxfunctions.splitfilename(i)  # extrayendo el nombre del archivo sin la extensión
            dir =pathname + '/' + name
            os.mkdir(dir)

            print('Generating crops for:', i)
            img = Image.open(imgPath + '/' + i)
            assert 3 * squareSize <= img.width * img.height
            # --------------------------
            # Obtencion recorte central aleatorio----------------------
            # Generando cuadrado dentral aleatorio
            # Genero un numero aleatorio que se le sumara o restara a las coordenadas centrales según una orientacion aleatoria
            randX = random.uniform(1, movementPixels)
            randY = random.uniform(1, movementPixels)

            randomOrientation = random.randint(0,
                                               8)  # Genero aleatoriamente hacia donde quiero que vaya más o menos el recorte central
            print('Orientation:', randomOrientation)

            # El recorte central se desplaza hacia la izquierda
            if randomOrientation == 1:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (
                                    img.width / 2) - randX  # xDist e yDist serán las varaibles para luego calcular los nuevos centros, no deben sobreescribirse con xCenter e yCenter
                yDist = (img.height / 2)

            # El recorte central se desplaza hacia la diagonal superior izquierda
            if randomOrientation == 2:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) - randX
                yDist = (img.height / 2) - randY

            # El recorte central se deplaza hacia arriba
            if randomOrientation == 3:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2)
                yDist = (img.height / 2) - randY

            # El recorte central se desplaza a la diagonal superior derecha
            if randomOrientation == 4:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) + randX
                yDist = (img.height / 2) - randY

            # El recorte central se desplaza a la derecha
            if randomOrientation == 5:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) + randX
                yDist = (img.height / 2) - randY

            # El recorte central se desplaza a la exquina inferior derecha
            if randomOrientation == 6:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) + randX
                yDist = (img.height / 2) + randY

            # El recorte central se desplaza hacia abajo
            if randomOrientation == 7:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2)
                yDist = (img.height / 2) + randY

            # El recorte central se desplaza a la diagonal inferior izquierda
            if randomOrientation == 8:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) - randX
                yDist = (img.height / 2) + randY

            # El recorte central se queda centrado
            if randomOrientation == 0:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2)
                yDist = (img.height / 2)

            # Cortamos y guardamos el cuadrado central

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            centralSquareCropped = img.crop(croppingMask)

            # centralSquareCropped.show('0')

            centralSquareCropped.save(pathname + '/' + name + '/c.jpg')

            # OBTENER EL RESTO DE RECORTES---------------------------------------------------------------------------------------------------

            # Obtención del recorte 0 (izquierda)-----------------------------------------
            # Distancia despecto del cuadradito central
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)  # Asi se aleja o acerca más del recorte del centro
            # Coordenadas respecto del central
            xCenter = xDist - centralSquareDistance
            yCenter = yDist + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            leftSquareCropped = img.crop(croppingMask)

            # leftSquareCropped.show('img1')

            leftSquareCropped.save(pathname + '/' + name + '/0.jpg')

            # Obtención del recorte 2 (arriba)-------------------------------------
            # Distancia respecto al cuadrado central
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)
            # Coordenadas respecto del central
            xCenter = xDist + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist - centralSquareDistance

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            upSquareCropped = img.crop(croppingMask)

            # upSquareCropped.show('img3')

            upSquareCropped.save(pathname + '/' + name + '/2.jpg')

            # Obtencion del recorte 6 (abajo)
            # Distancia respecto al cuadrado central
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)
            # Coordenadas respecto del central
            xCenter = xDist + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist + centralSquareDistance

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            downSquareCropped = img.crop(croppingMask)

            # downSquareCropped.show('img7')

            downSquareCropped.save(pathname + '/' + name + '/6.jpg')

            # Obtencion del recorte 4 (derecha)
            # Distancia despecto del cuadradito central
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)  # Asi se aleja o acerca más del recorte del centro
            # Coordenadas respecto del central
            xCenter = xDist + centralSquareDistance
            yCenter = yDist + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            rightSquareCropped = img.crop(croppingMask)

            # rightSquareCropped.show('img5')

            rightSquareCropped.save(pathname + '/' + name + '/4.jpg')

            # OBTENCION DE LAS DIAGONALES----------------------------------------------------------------------------------

            # Superior izquierda (1)
            # Coordenadas respecto del cuadrado central
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)
            # Coordenadas respecto del central
            xCenter = xDist - centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist - centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            leftUpSquareCropped = img.crop(croppingMask)

            leftUpSquareCropped.save(pathname + '/' + name + '/1.jpg')

            # Superior derecha (3)
            # Coordenadas respecto del cuadrado central
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)
            # Coordenadas respecto del central
            xCenter = xDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist - centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            rightUpSquareCropped = img.crop(croppingMask)

            rightUpSquareCropped.save(pathname + '/' + name + '/3.jpg')

            # Inferior izquierda (7)

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)
            # Coordenadas respecto del central
            xCenter = xDist - centralSquareDistance - float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            leftDownSquareCropped = img.crop(croppingMask)

            leftDownSquareCropped.save(pathname + '/' + name + '/7.jpg')

            # Inferior derecha(5)
            # Coordenadas respecto del cuadrado central
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)
            # Coordenadas respecto del central
            xCenter = xDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            rightDownSquareCropped = img.crop(croppingMask)

            rightDownSquareCropped.save(pathname + '/' + name + '/5.jpg')