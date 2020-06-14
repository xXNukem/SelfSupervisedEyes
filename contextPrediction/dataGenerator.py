import importlib.util
spec = importlib.util.spec_from_file_location("imgTools.py", "../DatasetCreation/imgTools.py")
imgTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imgTools)
import tensorflow.keras as keras
import cv2
import random
import numpy as np


#params inicialization
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgList, list_IDs,datagen,batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, normalize=True,downsampling=True, dataAugmentation=True,
                 downsamplingPercent=65,rgbToGray=True):
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
        K = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        resizer=imgTools.imgTools() #Get resizer
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            center,match,label=self.all_samples[ID]
            transformedC=cv2.imread(center)
            transformedM=cv2.imread(match)
            #Normalize input imgs
            if self.normalize == True:
                cv2.normalize(transformedC, transformedC, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(transformedM, transformedM, 0, 255, cv2.NORM_MINMAX)
            #Data augmentation
            if self.dataAugmentation == True:
                activation=random.randint(0,10)
                if activation > 1:
                    apply = random.randint(0,16)
                    transform = self.datagen.get_random_transform(self.dim,
                                                                  seed=random.seed(5))
                    if apply <= 5:
                        transformedC=self.datagen.apply_transform(transformedC,
                                                                  transform)
                    if apply >= 5 and apply <=11:
                        transformedM = self.datagen.apply_transform(transformedM,
                                                                    transform)
                    if apply >= 10:
                        transform = self.datagen.get_random_transform(self.dim,
                                                                      seed=random.seed(5))
                        transformedC = self.datagen.apply_transform(transformedC,
                                                                    transform)
                        transform = self.datagen.get_random_transform(self.dim,
                                                                      seed=random.seed(5))
                        transformedM = self.datagen.apply_transform(transformedM,
                                                                    transform)
            #Downsampling and upsamplig
            if self.downsampling == True:
                assert self.downsamplingPercent>1 and self.downsamplingPercent<100
                width,height=self.dim
                activation=random.randint(0,10)
                if activation > 1:
                    downsamplingWidth=(width*self.downsamplingPercent)/100
                    downsamplingHeight=(height*self.downsamplingPercent)/100
                    apply = random.randint(0,16)
                    if apply <= 5:
                        #downsampling
                        transformedC=resizer.image_resize(transformedC,
                                                          int(width-downsamplingWidth),
                                                          int(height-downsamplingHeight))
                        #upsampling
                        transformedC = resizer.image_resize(transformedC, width, height)
                    if apply >= 5 and apply <= 11:
                        # downsampling
                        transformedM = resizer.image_resize(transformedM,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformedM = resizer.image_resize(transformedM, width, height)
                    if apply >= 10:
                        # downsampling
                        transformedM = resizer.image_resize(transformedM,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformedM = resizer.image_resize(transformedM, width, height)
                        # downsampling
                        transformedC = resizer.image_resize(transformedC,
                                                            int(width - downsamplingWidth),
                                                            int(height - downsamplingHeight))
                        # upsampling
                        transformedC = resizer.image_resize(transformedC, width, height)
            # random rgb2gray
            if self.rgbToGray == True:
                activation = random.randint(0,10)
                if activation > 1:
                    apply = random.randint(0,16)
                    if apply <= 5:
                        transformedC = cv2.cvtColor(transformedC, cv2.COLOR_BGR2GRAY)
                        transformedC = cv2.cvtColor(transformedC, cv2.COLOR_GRAY2RGB)
                    if apply >= 5 and apply <= 11:
                        transformedM = cv2.cvtColor(transformedM, cv2.COLOR_BGR2GRAY)
                        transformedM = cv2.cvtColor(transformedM, cv2.COLOR_GRAY2RGB)
                    if apply >= 10:
                        transformedM = cv2.cvtColor(transformedM, cv2.COLOR_BGR2GRAY)
                        transformedM = cv2.cvtColor(transformedM, cv2.COLOR_GRAY2RGB)
                        transformedC = cv2.cvtColor(transformedC, cv2.COLOR_BGR2GRAY)
                        transformedC = cv2.cvtColor(transformedC, cv2.COLOR_GRAY2RGB)


            X[i,] = transformedC
            K[i,] = transformedM
            y[i]=int(label)

        return [X,K], keras.utils.to_categorical(y, num_classes=self.n_classes)
