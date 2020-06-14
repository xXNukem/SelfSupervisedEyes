import importlib.util
spec = importlib.util.spec_from_file_location("imgTools.py", "../DatasetCreation/imgTools.py")
imgTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imgTools)
from os import listdir
import cv2
import os
from random import choice

class rotation:

    #generate a dataset with rotated imgs
    """
    Its recomended to have the dataset preprocessed before this.
    You dont have to resize imgs before this because all imgs can be
    automatically resized in the Keras DataGenerator.
    """
    def generateDataset(self,imgPath,pathname):

        tools = imgTools.imgTools()
        if os.path.exists(pathname) == False:
            os.mkdir(pathname)
            os.mkdir(pathname + '/' + '0')
            os.mkdir(pathname + '/' + '1')
            os.mkdir(pathname + '/' + '2')
        imgs = listdir(imgPath)

        for img in imgs:
            name, ext = tools.splitfilename(img)
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

