import matplotlib.pyplot as plt
import pydicom
import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import numpy as np
import random
from os import listdir
import auxfunctions

class readIMG:

    def readDCMimage(self,imgName):
        dataset = pydicom.dcmread(imgName)

        print('------IMG Info--------')

        print("Filename->", imgName)
        print("Storage type->", dataset.SOPClassUID)
        print("Patient data:")
        print("Patient's name...:", dataset.PatientName)
        print("Patient id.......:", dataset.PatientID)
        print("Modality.........:", dataset.Modality)
        print("Study Date.......:", dataset.StudyDate)

        if 'PixelData' in dataset:
            rows = int(dataset.Rows)
            cols = int(dataset.Columns)
            print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
                rows=rows, cols=cols, size=len(dataset.PixelData)))
            if 'PixelSpacing' in dataset:
                print("Pixel spacing....:", dataset.PixelSpacing)

        print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

        plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
        plt.show()
#-----------------------------------------------------------------------------------------------------------------------

    def image_resize(self,image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    #------------------------------------------------------------------------------------------------------------

    def readDCMdataset(self,DCMPath):
        lstFilesDCM=[]
        for dirName,subDirList,fileList in os.walk(DCMPath):
            for fileName in fileList:
                lstFilesDCM.append(os.path.join(dirName,fileName))

        saveNumber=0
        if os.path.exists('./converted')==False:
            os.mkdir('./converted')
        for files in lstFilesDCM:
            dataset = pydicom.dcmread(files)
            Image.fromarray(dataset.pixel_array).save("./converted/img_" + str(saveNumber) + ".jpg")
            originalIMG = cv2.imread("./converted/img_" + str(saveNumber) + ".jpg")
            resizedIMG = self.image_resize(originalIMG, width=240, height=240)
            cv2.imwrite("./converted/img_" + str(saveNumber) + ".jpg", resizedIMG)
            saveNumber = saveNumber + 1

        print('Dataset transformed to JPG images into ./converted folder')
#------------------------------------------------------------------------------------------------------
    def generateDataset(self,imgPath,sqSize,sqPercent):

        if os.path.exists('./dataset')==False:
            os.mkdir('./dataset')

        imgDir=listdir(imgPath)
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
            name,exte=auxfunctions.splitfilename(i)#extrayendo el nombre del archivo sin la extensión
            dir='./dataset/'+name
            os.mkdir(dir)

            print('Generating crops for:',i)
            img = Image.open(imgPath+'/'+i)
            assert 3 * squareSize <= img.width * img.height
            # --------------------------
            # Obtencion recorte central aleatorio----------------------
            # Generando cuadrado dentral aleatorio
            # Genero un numero aleatorio que se le sumara o restara a las coordenadas centrales según una orientacion aleatoria
            randX = random.uniform(1, movementPixels)
            randY = random.uniform(1, movementPixels)

            randomOrientation = random.randint(0, 8)  # Genero aleatoriamente hacia donde quiero que vaya más o menos el recorte central
            print('Orientation:', randomOrientation)

            # El recorte central se desplaza hacia la izquierda
            if randomOrientation == 1:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) - randX  # xDist e yDist serán las varaibles para luego calcular los nuevos centros, no deben sobreescribirse con xCenter e yCenter
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

            centralSquareCropped.save('./dataset/'+name+'/c.jpg')

            # OBTENER EL RESTO DE RECORTES---------------------------------------------------------------------------------------------------

            # Obtención del recorte 0 (izquierda)-----------------------------------------
            # Distancia despecto del cuadradito central
            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            leftSquareCropped.save('./dataset/'+name+'/0.jpg')

            # Obtención del recorte 2 (arriba)-------------------------------------
            # Distancia respecto al cuadrado central
            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            upSquareCropped.save('./dataset/'+name+'/2.jpg')

            # Obtencion del recorte 6 (abajo)
            # Distancia respecto al cuadrado central
            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            downSquareCropped.save('./dataset/'+name+'/6.jpg')

            # Obtencion del recorte 4 (derecha)
            # Distancia despecto del cuadradito central
            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            rightSquareCropped.save('./dataset/'+name+'/4.jpg')

            # OBTENCION DE LAS DIAGONALES----------------------------------------------------------------------------------

            # Superior izquierda (1)
            # Coordenadas respecto del cuadrado central
            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            leftUpSquareCropped.save('./dataset/'+name+'/1.jpg')

            # Superior derecha (3)
            # Coordenadas respecto del cuadrado central
            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            rightUpSquareCropped.save('./dataset/'+name+'/3.jpg')

            # Inferior izquierda (7)

            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            leftDownSquareCropped.save('./dataset/'+name+'/7.jpg')

            # Inferior derecha(5)
            # Coordenadas respecto del cuadrado central
            centralSquareDistance = (squareSize / 2) + float((random.randrange(int(movementPixels)) + movementPixels)) + (
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

            rightDownSquareCropped.save('./dataset/'+name+'/5.jpg')