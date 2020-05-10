from os import listdir
import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import img_to_array
import auxfunctions
import pickle
from PIL import Image

class jiggsawFunctions:

    def generateDataset(self,imgPath, sqSize, sqPercent, pathname):

        if os.path.exists(pathname) == False:
            os.mkdir(pathname)

        