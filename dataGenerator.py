import numpy as np
import tensorflow.keras as keras
import cv2
import random
import auxfunctions
import readDCM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgList, list_IDs,downsamplingPercent, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, normalize=True,downsampling=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.all_samples=imgList
        self.normalize=normalize
        self.downsampling=downsampling
        self.downsamplingPercent=downsamplingPercent
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        K = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        #Obtener media y desviacion tipica del fichero
        mean, std = auxfunctions.getMeanStd()
        resizer=readDCM.readIMG() #objeto de la clase readIMG para resizar las imagenes
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            center,match,label=self.all_samples[ID]
            transformedC=cv2.imread(center)
            transformedM=cv2.imread(match)

            #Normalizar imagenes entre 0 y 1
            if self.normalize == True:
                cv2.normalize(transformedC, transformedC, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformedM, transformedM, 0, 255, cv2.NORM_MINMAX)
            #Realizar downsampling y upsampling a la pareja
            if self.downsampling==True:
                assert self.downsamplingPercent>1 and self.downsamplingPercent<100
                width,height=self.dim
                activation=random.uniform(-50,50)
                if activation < 0:
                    downsamplingWidth=(width*self.downsamplingPercent)/100
                    downsamplingHeight=(height*self.downsamplingPercent)/100
                    #downsampling
                    #transformedC=resizer.image_resize(transformedC,int(width-downsamplingWidth),int(height-downsamplingHeight))
                    transformedM=resizer.image_resize(transformedM,int(width-downsamplingWidth),int(height-downsamplingHeight))
                    #upsampling
                    #transformedC = resizer.image_resize(transformedC, width, height)
                    transformedM = resizer.image_resize(transformedM, width, height)

            X[i,] = transformedC
            K[i,] = transformedM
            y[i]=int(label)

        return [X,K], keras.utils.to_categorical(y, num_classes=self.n_classes)

"""
params = {'dim': (96,96),
          'batch_size':64,
          'n_classes': 8,
          'n_channels': 3,
          'shuffle': True,
          'normalize': True,
          'downsampling':True,
          'downsamplingPercent':60}

obj=DataGenerator([('./drRafael2/img_0/c.jpg','./drRafael2/img_0/1.jpg','1'),
                   ('./drRafael2/img_0/c.jpg','./drRafael2/img_0/3.jpg','3'),
                   ('./drRafael2/img_0/c.jpg','./drRafael2/img_0/4.jpg','4'),
                   ('./drRafael2/img_0/c.jpg','./drRafael2/img_0/6.jpg','6')],[0,1,2,3],**params)

x,y=obj.__getitem__(0)

print(x)
"""

