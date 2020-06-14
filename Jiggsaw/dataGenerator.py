import importlib.util
spec = importlib.util.spec_from_file_location("imgTools.py", "../DatasetCreation/imgTools.py")
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

        resizer=imgTools.imgTools() #get resizer

        for i, ID in enumerate(list_IDs_temp):
            img1,img2,img3,img4,label=self.all_samples[ID]
            transformed1 = cv2.imread(img1)
            transformed2 = cv2.imread(img2)
            transformed3 = cv2.imread(img3)
            transformed4 = cv2.imread(img4)

            #Normalize imgs
            if self.normalize == True:
                cv2.normalize(transformed1, transformed1, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformed2, transformed2, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformed3, transformed3, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformed4, transformed4, 0, 255, cv2.NORM_MINMAX)
            #Data augmentation
            if self.dataAugmentation == True:
                activation=random.randint(2,10)
                if activation > 1:
                    apply = random.randint(1,25)
                    transform = self.datagen.get_random_transform(self.dim,
                                                                  seed=random.seed(5))
                    if apply <= 5:
                        transformed1=self.datagen.apply_transform(transformed1,
                                                                  transform)
                    if apply >= 5 and apply <=10:
                        transformed2 = self.datagen.apply_transform(transformed2,
                                                                    transform)
                    if apply >= 10 and apply <= 15:
                        transformed3 = self.datagen.apply_transform(transformed3,
                                                                    transform)
                    if apply >=15 and apply <=20:
                        transformed4 = self.datagen.apply_transform(transformed4,
                                                                    transform)
                    if apply >=20:
                        transform = self.datagen.get_random_transform(self.dim,
                                                                      seed=random.seed(5))
                        transformed1 = self.datagen.apply_transform(transformed1,
                                                                    transform)
                        transform = self.datagen.get_random_transform(self.dim,
                                                                      seed=random.seed(5))
                        transformed2 = self.datagen.apply_transform(transformed2,
                                                                    transform)
                        transform = self.datagen.get_random_transform(self.dim,
                                                                      seed=random.seed(5))
                        transformed3 = self.datagen.apply_transform(transformed3,
                                                                    transform)
                        transform = self.datagen.get_random_transform(self.dim,
                                                                      seed=random.seed(5))
                        transformed4 = self.datagen.apply_transform(transformed4,
                                                                    transform)
            #Random upsampling and downsampling to some patches
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
                        transformed1=resizer.image_resize(transformed1,
                                                          int(width-downsamplingWidth),
                                                          int(height-downsamplingHeight))
                        #upsampling
                        transformed1 = resizer.image_resize(transformed1,
                                                            width,
                                                            height)
                    if apply >= 5 and apply <= 10:
                        # downsampling
                        transformed2 = resizer.image_resize(transformed2,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed2 = resizer.image_resize(transformed2,
                                                            width,
                                                            height)
                    if apply >= 10 and apply <=15:
                        # downsampling
                        transformed3 = resizer.image_resize(transformed3,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed3 = resizer.image_resize(transformed3,
                                                            width,
                                                            height)
                    if apply >= 15 and apply <=20:
                        # downsampling
                        transformed4 = resizer.image_resize(transformed4,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed4 = resizer.image_resize(transformed4,
                                                            width,
                                                            height)
                    if apply >=20:

                        # downsampling
                        transformed1 = resizer.image_resize(transformed1,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed1 = resizer.image_resize(transformed1,
                                                            width,
                                                            height)
                        # downsampling
                        transformed2 = resizer.image_resize(transformed2,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed2 = resizer.image_resize(transformed2,
                                                            width,
                                                            height)
                        # downsampling
                        transformed3 = resizer.image_resize(transformed3,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed3 = resizer.image_resize(transformed3,
                                                            width,
                                                            height)
                        # downsampling
                        transformed4 = resizer.image_resize(transformed4,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformed4 = resizer.image_resize(transformed4,
                                                            width,
                                                            height)

            # random to grayscale
            if self.rgbToGray == True:
                activation = random.randint(0,10)
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


            X[i,] = transformed1
            Y[i,] = transformed2
            Z[i,] = transformed3
            K[i,] = transformed4
            y[i]=int(label)

        return [X,Y,Z,K], keras.utils.to_categorical(y, num_classes=self.n_classes)
