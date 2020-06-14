import os
import pandas as pd
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class dataBalancing:

    #Categorize imgs with the csv file
    #csvPath: Path of the required csv file
    #imgPath: Path of the uncategorized imgs
    #destinationPath: Path for saving categorized imgs
    #nCategories: Number of classes given in the problem, this allow to use this functions in other problems
    def categorizeDataset(self,csvPath,imgPath,destinationPath,nCategories):

        if os.path.exists(destinationPath) == False:
            os.mkdir(destinationPath)
            for i in range(0,nCategories):
                os.mkdir(destinationPath+'/'+str(i))

        df=pd.read_csv(csvPath)

        for i in range(0,len(df)):
            imagen=df['image'][i]
            nivel=df['level'][i]
            print(imagen)
            print(nivel)
            if os.path.exists(imgPath+'/'+imagen+'.jpeg'):
                img=cv2.imread(imgPath+'/'+imagen+'.jpeg')
                cv2.imwrite(destinationPath+'/'+str(nivel)+'/'+imagen+'.jpeg',img)
            else:
                print('Img',imagen,'not found')

    #decrease samples on a directory after categorize all imgs
    #path: Directory to undersample
    #nSamples: Total samples alowed
    def undersampleCategory(self,path,nSamples):
        print('Undersampling...')
        while len(os.listdir(path)) >=nSamples:
                list = os.listdir(path)
                rand = random.randint(0, len(list)-1)
                os.remove(path+'/'+list[rand])


    #increase samples on a directory with dataaugmentation
    # path: Directory to undersample
    # nSamples: Total samples alowed
    def oversampleCategory(self,path,nSamples):
        print('Oversampling...')
        datagen = ImageDataGenerator(#rescale=1./255.0,
                                    zoom_range=0.2,
                                     width_shift_range=[-1.5,1.5],
                                     height_shift_range=[-1.5,1.5],
                                     rotation_range=1.5,
                                     shear_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=False,
                                     brightness_range=[0.95,1.05],
                                    channel_shift_range=1.2
                                     )

        c=0
        d=0
        initialList=os.listdir(path)
        while len(os.listdir(path)) <nSamples:
                img = cv2.imread(path+'/'+initialList[c])

                shape = img.shape
                transform = datagen.get_random_transform(shape, seed=random.seed())
                img = datagen.apply_transform(img, transform)

                grey = random.randint(0, 10) #some images will be converted to grayscale
                if grey > 9:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                cv2.imwrite(path+'/'+str(d)+initialList[c], img)

                c=c+1
                d=d+1
                if c>len(initialList)-1:
                    c=0
