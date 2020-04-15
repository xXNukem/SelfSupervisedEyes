from os import listdir,path
import cv2
import numpy as np
import os

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

#Carga una lista de tuplas con las rutas a las imágenes emparejadas
def loadimgspath(path):

    X=[]

    for i in listdir(path):
        imgs=listdir(path+'/'+i)
        for j in ['0','1','2','3','4','5','6','7']:
                #Centro                 cuadrado                    etiqueta
            aux=(path+'/'+i+'/'+imgs[8],path+'/'+i+'/'+imgs[int(j)],j)
            X.append(aux)

    return X
