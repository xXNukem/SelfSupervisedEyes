import importlib.util

spec = importlib.util.spec_from_file_location("imgTools.py", "../main/imgTools.py")
imgTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imgTools)

import tensorflow.keras as keras
import cv2
import random
import numpy as np
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgList, list_IDs,datagen,batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, normalize=True,downsampling=True, dataAugmentation=True,downsamplingPercent=65,rgbToGray=True):
        'Initialization'
        self.dim = dim
        self.imgList=imgList
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.all_samples=imgList
        self.normalize=normalize
        self.downsampling=downsampling
        self.downsamplingPercent=downsamplingPercent
        self.dataAugmentation=dataAugmentation
        self.rgbToGray=rgbToGray
        self.datagen=datagen
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

    def getClasses(self):
        classes=[]
        for i in range(0,len(self.imgList)):
            img,label=self.imgList[i]
            classes.append(int(label))
        return classes,len(classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        #Obtener media y desviacion tipica del fichero

        resizer=imgTools.imgTools() #objeto de la clase readIMG para resizar las imagenes
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img, label=self.all_samples[ID]
            #mean, std = auxfunctions.getMeanStd()
            transformedImg=cv2.imread(img)
            #Normalizar imagenes
            if self.normalize == True:
                cv2.normalize(transformedImg, transformedImg, 0, 255, cv2.NORM_MINMAX)
            #Data augmentation
            if self.dataAugmentation == True:
                activation=random.randint(2,10)
                if activation > 1:
                    transform = self.datagen.get_random_transform(self.dim, seed=random.seed(5))
                    transformedImg=self.datagen.apply_transform(transformedImg,transform)

            #Realizar downsampling y upsampling a la pareja
            if self.downsampling == True:
                assert self.downsamplingPercent>1 and self.downsamplingPercent<100
                width,height=self.dim
                activation=random.randint(0,10)
                if activation > 1:
                    downsamplingWidth=(width*self.downsamplingPercent)/100
                    downsamplingHeight=(height*self.downsamplingPercent)/100
                    #downsampling
                    transformedImg=resizer.image_resize(transformedImg,int(width-downsamplingWidth),int(height-downsamplingHeight))
                    #upsampling
                    transformedImg = resizer.image_resize(transformedImg, width, height)

            # Pasar aleatoriamente a escala de grises
            if self.rgbToGray == True:
                activation = random.randint(0,10)
                if activation > 1:
                    transformedImg = cv2.cvtColor(transformedImg, cv2.COLOR_BGR2GRAY)
                    transformedImg = cv2.cvtColor(transformedImg, cv2.COLOR_GRAY2RGB)

            X[i,] = transformedImg
            y[i]=int(label)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


"""
datagen = ImageDataGenerator(#rescale=1.0/255,
                            #zoom_range=[-2, 2],
                             #width_shift_range=[-25, 25],
                             #height_shift_range=[-25, 25],
                             #rotation_range=40,
                             #shear_range=40,
                             #horizontal_flip=True,
                             #vertical_flip=True,
                             brightness_range=[0.98,1.05],
                             #featurewise_center=True,
                             #samplewise_center=True,
                             channel_shift_range=0.95
                             )


params = {'dim': (96,96),
          'batch_size':2,
          'n_classes': 8,
          'n_channels': 3,
          'shuffle': True,
          'normalize': True,
          'downsampling':True,
          'downsamplingPercent':80,
          'dataAugmentation':True,
            'rgbToGray':False,
          'datagen':datagen
            }

obj=DataGenerator([('./drRafael2/img_0/c.jpg','./drRafael2/img_0/1.jpg','1'),
                   ('./drRafael2/img_0/c.jpg','./drRafael2/img_0/3.jpg','3'),
                   ('./drRafael2/img_0/c.jpg','./drRafael2/img_0/4.jpg','4'),
                   ('./drRafael2/img_0/c.jpg','./drRafael2/img_0/6.jpg','6')],[0,1,2,3],**params)

x,y=obj.__getitem__(1)

transform =datagen.get_random_transform((96,96), seed=random.seed(5))
transform['brightness']=0.999
#transform['channel_shift_intensity']=5
img=cv2.imread('./drRafael2/img_0/c.jpg')
#img=img/255.0
cv2.imshow('sdf',img)
img2=datagen.apply_transform(img,transform)
#img2=img2/255.0
cv2.imshow('dfa',img2)

cv2.waitKey()
"""
