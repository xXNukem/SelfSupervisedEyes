import importlib.util
spec = importlib.util.spec_from_file_location("imgTools.py", "../DatasetCreation/imgTools.py")
imgTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imgTools)
from os import listdir
import os
import random
import pickle
from PIL import Image

class contextPrediction:

    # Create a list with imgs path
    #path: path to a directory with imgs
    def loadimgspath(self,path):

        X = []

        for i in listdir(path):
            imgs = listdir(path + '/' + i)
            for j in ['0', '1', '2', '3', '4', '5', '6', '7']:
                # Center                                 Pair                               Class
                aux = (path + '/' + i + '/' + imgs[8], path + '/' + i + '/' + imgs[int(j)], j)
                print('Pair created: ', aux)
                X.append(aux)

        return X

    # Split data in train/validation
    #imgList: list created in function 'loadimgspath'
    #percent: percentaje for validation split
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

    # read train/validation splits
    def getTrainValidationSplits(self):
        with open('train.pickle', 'rb') as file:
            train = pickle.load(file)

        with open('validation.pickle', 'rb') as file:
            validation = pickle.load(file)

        return train, validation #return train and validation splits

    #Generate the dataset
    """
    imgPath: Path to a directory with imgs
    sqSize: Size of the patch to crop
    sqPercent: Variation in the distance between patches
    pathname: Destination Folder
    
    Its recomended to resize and preprocess your imgs before this
    """
    def generateDataset(self, imgPath, sqSize, sqPercent, pathname):

        if os.path.exists(pathname) == False:
            os.mkdir(pathname)

        imgDir = listdir(imgPath)
        # Square Size
        squareSize = sqSize
        # Moving pixels
        squareSizePercent = sqPercent
        # Maximum moving pixels
        movementPixels = (squareSize * squareSizePercent) / 100
        assert squareSizePercent >= 1 or squareSizePercent <= 100

        tools = imgTools.imgTools()
        for i in imgDir:
            name, exte = tools.splitfilename(i)  # get file name
            dir =pathname + '/' + name
            os.mkdir(dir)

            print('Generating crops for:', i)
            img = Image.open(imgPath + '/' + i)
            assert 3 * squareSize <= img.width * img.height
            # --------------------------
            # Generating a random positioned central patch
            randX = random.uniform(1, movementPixels)
            randY = random.uniform(1, movementPixels)

            randomOrientation = random.randint(0,
                                               8)
            print('Orientation:', randomOrientation)

            # left orientation
            if randomOrientation == 1:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (
                                    img.width / 2) - randX  # xDist, yDist will calculate new centers after this
                yDist = (img.height / 2)

            # up left orientation
            if randomOrientation == 2:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) - randX
                yDist = (img.height / 2) - randY

            # up orientation
            if randomOrientation == 3:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2)
                yDist = (img.height / 2) - randY

            # up right orientation
            if randomOrientation == 4:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) + randX
                yDist = (img.height / 2) - randY

            # right orientation
            if randomOrientation == 5:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) + randX
                yDist = (img.height / 2) - randY

            # down right orientation
            if randomOrientation == 6:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) + randX
                yDist = (img.height / 2) + randY

            # down orientation
            if randomOrientation == 7:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2)
                yDist = (img.height / 2) + randY

            # down left orientation
            if randomOrientation == 8:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2) - randX
                yDist = (img.height / 2) + randY

            # central square remains centered
            if randomOrientation == 0:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)
                xDist = (img.width / 2)
                yDist = (img.height / 2)

            # cutting and saving central patch

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            centralSquareCropped = img.crop(croppingMask)

            centralSquareCropped.save(pathname + '/' + name + '/c.jpg')

            # Get Remain Patches---------------------------------------------------------------

            # Patch 0 left-----------------------------------------
            # distance from central patch
            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)  # more or less distance from central patch
            # New X/Y
            xCenter = xDist - centralSquareDistance
            yCenter = yDist + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            leftSquareCropped = img.crop(croppingMask)


            leftSquareCropped.save(pathname + '/' + name + '/0.jpg')

            # Patch 2 up -------------------------------------

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)

            xCenter = xDist + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist - centralSquareDistance

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            upSquareCropped = img.crop(croppingMask)

            upSquareCropped.save(pathname + '/' + name + '/2.jpg')

            # Patch 6 down

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)

            xCenter = xDist + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist + centralSquareDistance

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            downSquareCropped = img.crop(croppingMask)


            downSquareCropped.save(pathname + '/' + name + '/6.jpg')

            # Patch 4 right

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)

            xCenter = xDist + centralSquareDistance
            yCenter = yDist + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            rightSquareCropped = img.crop(croppingMask)

            rightSquareCropped.save(pathname + '/' + name + '/4.jpg')

            # Getting diagonal patches----------------------------------------------------------------------------------

            # up left 1

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)

            xCenter = xDist - centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist - centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            leftUpSquareCropped = img.crop(croppingMask)

            leftUpSquareCropped.save(pathname + '/' + name + '/1.jpg')

            # up right 3

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)

            xCenter = xDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist - centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            rightUpSquareCropped = img.crop(croppingMask)

            rightUpSquareCropped.save(pathname + '/' + name + '/3.jpg')

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)

            xCenter = xDist - centralSquareDistance - float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            leftDownSquareCropped = img.crop(croppingMask)

            leftDownSquareCropped.save(pathname + '/' + name + '/7.jpg')

            # down right 5

            centralSquareDistance = (squareSize / 2) + float(
                (random.randrange(int(movementPixels)) + movementPixels)) + (
                                            squareSize / 2)

            xCenter = xDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))
            yCenter = yDist + centralSquareDistance + float(random.uniform(-movementPixels, movementPixels))

            x1 = xCenter - (squareSize / 2)
            y1 = yCenter - (squareSize / 2)
            x2 = xCenter + (squareSize / 2)
            y2 = yCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            rightDownSquareCropped = img.crop(croppingMask)

            rightDownSquareCropped.save(pathname + '/' + name + '/5.jpg')