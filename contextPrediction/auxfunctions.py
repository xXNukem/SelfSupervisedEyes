from os import listdir,path
import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import img_to_array
import csv
import pickle
#Obtiene el nombre y la extension de un archivo
def splitfilename(filename):
    sname=""
    sext=""
    i=filename.rfind(".")
    if(i!=0):
        n=len(filename)
        j=n-i-1
        sname=filename[0:i]
        sext=filename[-j:]
    return sname, sext

