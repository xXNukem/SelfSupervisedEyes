import importlib.util

spec = importlib.util.spec_from_file_location("imgTools.py", "../main/imgTools.py")
imgTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imgTools)
import tensorflow.keras as keras
import cv2
import random
import numpy as np
import jiggsawFunctions

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgList, list_IDs,datagen,batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, normalize=True,downsampling=True, dataAugmentation=True,downsamplingPercent=65,rgbToGray=True):
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

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))
        Z = np.empty((self.batch_size, *self.dim, self.n_channels))
        K = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        #Obtener media y desviacion tipica del fichero
        functions=jiggsawFunctions.jiggsaw()
        mean,std=functions.getMeanStd()
        resizer=imgTools.imgTools() #objeto de la clase readIMG para resizar las imagenes
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img1,img2,img3,img4,label=self.all_samples[ID]
            transformed1 = cv2.imread(img1)
            transformed2 = cv2.imread(img2)
            transformed3 = cv2.imread(img3)
            transformed4 = cv2.imread(img4)

            #Normalizar imagenes
            if self.normalize == True:
                cv2.normalize(transformed1, transformed1, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformed2, transformed2, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformed3, transformed3, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformed4, transformed4, 0, 255, cv2.NORM_MINMAX)
            #Data augmentation
            if self.dataAugmentation == True:
                activation=random.randint(0,10)
                if activation > 1:
                    apply = random.randint(1,25)
                    transform = self.datagen.get_random_transform(self.dim, seed=random.seed(5))
                    if apply <= 5:
                        transformed1=self.datagen.apply_transform(transformed1,transform)
                    if apply >= 5 and apply <=10:
                        transformed2 = self.datagen.apply_transform(transformed2, transform)
                    if apply >= 10 and apply <= 15:
                        transformed3 = self.datagen.apply_transform(transformed3, transform)
                    if apply >=15 and apply <=20:
                        transformed4 = self.datagen.apply_transform(transformed4, transform)
                    if apply >=20:
                        transform = self.datagen.get_random_transform(self.dim, seed=random.seed(5))
                        transformed1 = self.datagen.apply_transform(transformed1, transform)
                        transform = self.datagen.get_random_transform(self.dim, seed=random.seed(5))
                        transformed2 = self.datagen.apply_transform(transformed2, transform)
                        transform = self.datagen.get_random_transform(self.dim, seed=random.seed(5))
                        transformed3 = self.datagen.apply_transform(transformed3, transform)
                        transform = self.datagen.get_random_transform(self.dim, seed=random.seed(5))
                        transformed4 = self.datagen.apply_transform(transformed4, transform)
            #Realizar downsampling y upsampling a la pareja
            if self.downsampling == True:
                assert self.downsamplingPercent>1 and self.downsamplingPercent<100
                width,height=self.dim
                activation=random.randint(0,10)
                if activation > 1:
                    downsamplingWidth=(width*self.downsamplingPercent)/100
                    downsamplingHeight=(height*self.downsamplingPercent)/100
                    apply = random.randint(1,25)
                    if apply <= 5:
                        #downsampling
                        transformed1=resizer.image_resize(transformed1,int(width-downsamplingWidth),int(height-downsamplingHeight))
                        #upsampling
                        transformed1 = resizer.image_resize(transformed1, width, height)
                    if apply >= 5 and apply <= 10:
                        # downsampling
                        transformed2 = resizer.image_resize(transformed2, int(width - downsamplingWidth),int(height - downsamplingHeight))
                        # upsampling
                        transformed2 = resizer.image_resize(transformed2, width, height)
                    if apply >= 10 and apply <=15:
                        # downsampling
                        transformed3 = resizer.image_resize(transformed3, int(width - downsamplingWidth),int(height - downsamplingHeight))
                        # upsampling
                        transformed3 = resizer.image_resize(transformed3, width, height)
                    if apply >= 15 and apply <=20:
                        # downsampling
                        transformed4 = resizer.image_resize(transformed4, int(width - downsamplingWidth),int(height - downsamplingHeight))
                        # upsampling
                        transformed4 = resizer.image_resize(transformed4, width, height)
                    if apply >=20:

                        # downsampling
                        transformed1 = resizer.image_resize(transformed1, int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed1 = resizer.image_resize(transformed1, width, height)
                        # downsampling
                        transformed2 = resizer.image_resize(transformed2, int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed2 = resizer.image_resize(transformed2, width, height)
                        # downsampling
                        transformed3 = resizer.image_resize(transformed3, int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed3 = resizer.image_resize(transformed3, width, height)
                        # downsampling
                        transformed4 = resizer.image_resize(transformed4, int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed4 = resizer.image_resize(transformed4, width, height)

            # Pasar aleatoriamente a escala de grises
            if self.rgbToGray == True:
                activation = random.randint(-5,10)
                if activation > 1:
                    apply = random.randint(1,25)
                    if apply <= 5:
                        transformed1 = cv2.cvtColor(transformed1, cv2.COLOR_BGR2GRAY)
                        transformed1 = cv2.cvtColor(transformed1, cv2.COLOR_GRAY2RGB)
                    if apply >= 5 and apply <= 10:
                        transformed2 = cv2.cvtColor(transformed2, cv2.COLOR_BGR2GRAY)
                        transformed2 = cv2.cvtColor(transformed2, cv2.COLOR_GRAY2RGB)
                    if apply >= 10 and apply <=15:
                        transformed3 = cv2.cvtColor(transformed3, cv2.COLOR_BGR2GRAY)
                        transformed3 = cv2.cvtColor(transformed3, cv2.COLOR_GRAY2RGB)
                    if apply >=15 and apply <=20:
                        transformed4 = cv2.cvtColor(transformed4, cv2.COLOR_BGR2GRAY)
                        transformed4 = cv2.cvtColor(transformed4, cv2.COLOR_GRAY2RGB)
                    if apply >=20:
                        transformed1 = cv2.cvtColor(transformed1, cv2.COLOR_BGR2GRAY)
                        transformed1 = cv2.cvtColor(transformed1, cv2.COLOR_GRAY2RGB)
                        transformed2 = cv2.cvtColor(transformed2, cv2.COLOR_BGR2GRAY)
                        transformed2 = cv2.cvtColor(transformed2, cv2.COLOR_GRAY2RGB)
                        transformed3 = cv2.cvtColor(transformed3, cv2.COLOR_BGR2GRAY)
                        transformed3 = cv2.cvtColor(transformed3, cv2.COLOR_GRAY2RGB)
                        transformed4 = cv2.cvtColor(transformed4, cv2.COLOR_BGR2GRAY)
                        transformed4 = cv2.cvtColor(transformed4, cv2.COLOR_GRAY2RGB)


            #aplicar media y std al terminar
            #transformed1 = transformed1 - mean
            #transformed1 = transformed1 / std
            #transformed2 = transformed2 - mean
            #transformed2 = transformed2 / std
            #transformed3 = transformed3 - mean
            #transformed3 = transformed3 / std
            #transformed4 = transformed4 - mean
            #transformed4 = transformed4 / std


            X[i,] = transformed1
            Y[i,] = transformed2
            Z[i,] = transformed3
            K[i,] = transformed4
            y[i]=int(label)

        return [X,Y,Z,K], keras.utils.to_categorical(y, num_classes=self.n_classes)


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
