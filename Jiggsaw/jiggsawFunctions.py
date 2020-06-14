import importlib.util
spec = importlib.util.spec_from_file_location("imgTools.py", "../DatasetCreation/imgTools.py")
imgTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imgTools)
from os import listdir
import os
import random
import pickle
import numpy as np
from itertools import permutations
from PIL import Image

class jiggsaw:

    #generate dataset, works like context prediction but with only 4 patchs
    """
    imgPath: Path to a directory with imgs
    sqSize: Size of the patch to crop
    sqPercent: Variation in the distance between patches
    pathname: Destination Folder

    Its recomended to resize and preprocess your imgs before this
    """
    def generateDataset(self,imgPath, sqSize, pathname,sqPercent):

        variationPercent=(sqSize*sqPercent)/100

        if os.path.exists(pathname) == False:
            os.mkdir(pathname)

        imgDir = listdir(imgPath)
        squareSize = sqSize
        tools = imgTools.imgTools()
        for i in imgDir:
            name, exte = tools.splitfilename(i)
            dir =pathname + '/' + name
            os.mkdir(dir)

            print('Generating crops for:', i)
            img = Image.open(imgPath + '/' + i)
            assert 3 * squareSize <= img.width * img.height

            randX = random.uniform(1, int(sqSize/5))
            randY = random.uniform(1, int(sqSize/5))

            randomOrientation = random.randint(0,
                                               8)
            print('Orientation:', randomOrientation)


            if randomOrientation == 1:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 2:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 3:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 4:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 5:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) - randY
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 6:
                xCenter = (img.width / 2) + randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 7:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 8:
                xCenter = (img.width / 2) - randX
                yCenter = (img.height / 2) + randY
                print('X:', xCenter, 'Y:', yCenter)

            if randomOrientation == 0:
                xCenter = (img.width / 2)
                yCenter = (img.height / 2)
                print('X:', xCenter, 'Y:', yCenter)


            # Patch 0 up left

            newXCenter = (xCenter - int(sqSize/2))+random.uniform(-variationPercent,variationPercent)
            newYCenter = (yCenter - int(sqSize/2))+random.uniform(-variationPercent,variationPercent)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            upLeftSquare = img.crop(croppingMask)

            upLeftSquare.save(pathname + '/' + name + '/0.jpg')

            # patch1 up right

            newXCenter = (xCenter + int(sqSize/2))+random.uniform(-variationPercent,variationPercent)
            newYCenter = (yCenter - int(sqSize/2))+random.uniform(-variationPercent,variationPercent)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            upRightSquare = img.crop(croppingMask)

            upRightSquare.save(pathname + '/' + name + '/1.jpg')

            # patch 2 down right

            newXCenter = (xCenter + int(sqSize/2))+random.uniform(-variationPercent,variationPercent)
            newYCenter = (yCenter + int(sqSize/2))+random.uniform(-variationPercent,variationPercent)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            downRightSquare = img.crop(croppingMask)

            downRightSquare.save(pathname + '/' + name + '/2.jpg')

            # patch 3 down left

            newXCenter = (xCenter - int(sqSize/2))+random.uniform(-variationPercent,variationPercent)
            newYCenter = (yCenter + int(sqSize/2))+random.uniform(-variationPercent,variationPercent)

            x1 = newXCenter - (squareSize / 2)
            y1 = newYCenter - (squareSize / 2)
            x2 = newXCenter + (squareSize / 2)
            y2 = newYCenter + (squareSize / 2)

            croppingMask = (x1, y1, x2, y2)

            downRightSquare = img.crop(croppingMask)

            downRightSquare.save(pathname + '/' + name + '/3.jpg')

    #get distance between permutations (coded by Manuel Jesus Marin)
    def mj_dist_perms(self,p1, p2):
        d = 0
        for i in range(len(p1)):
            if p1[i] != p2[i]:
                d += 1
        return d

    #get a permutation list with maximum hamming distance (coded by Manuel Jesus Marin)
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


    # load imgs path of all permutations
    def loadimgspath(self, path,maxDist):
        permutationList=self.getPermutationList(maxDist)
        X = []

        for i in listdir(path):
            imgs = listdir(path + '/' + i)
            for j in imgs:
               randomPerm=random.randint(0,len(permutationList)-1) #randomPerm is also the label
               perm=permutationList[randomPerm] #save the corresponging label for each permutation
               aux=(path + '/' + i+'/'+str(perm[0])+'.jpg',path + '/' + i+'/'+str(perm[1])+'.jpg',path +
                    '/' + i+'/'+str(perm[2])+'.jpg',path + '/' + i+'/'+str(perm[3])+'.jpg',str(randomPerm))
               print('Perm created: ',aux)
               X.append(aux)

        return X

        # Splits permutation list in train/validation
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

        # read train/validation files
    def getTrainValidationSplits(self):
        with open('train.pickle', 'rb') as file:
            train = pickle.load(file)

        with open('validation.pickle', 'rb') as file:
            validation = pickle.load(file)

        return train, validation
