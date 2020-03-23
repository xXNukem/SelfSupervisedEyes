import matplotlib.pyplot as plt
import pydicom
import os
from PIL import Image, ImageOps
import numpy as np
import cv2

class readIMG:

    def readDCMimage(self,imgName):
        dataset = pydicom.dcmread(imgName)

        print('------IMG Info--------')

        print("Filename->", imgName)
        print("Storage type->", dataset.SOPClassUID)
        print("Patient data:")
        print("Patient's name...:", dataset.PatientName)
        print("Patient id.......:", dataset.PatientID)
        print("Modality.........:", dataset.Modality)
        print("Study Date.......:", dataset.StudyDate)

        if 'PixelData' in dataset:
            rows = int(dataset.Rows)
            cols = int(dataset.Columns)
            print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
                rows=rows, cols=cols, size=len(dataset.PixelData)))
            if 'PixelSpacing' in dataset:
                print("Pixel spacing....:", dataset.PixelSpacing)

        print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

        plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
        plt.show()
#-----------------------------------------------------------------------------------------------------------------------

    def image_resize(self,image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    #------------------------------------------------------------------------------------------------------------

    def readDCMdataset(self,DCMPath):
        lstFilesDCM=[]
        for dirName,subDirList,fileList in os.walk(DCMPath):
            for fileName in fileList:
                lstFilesDCM.append(os.path.join(dirName,fileName))

        saveNumber=0
        if os.path.exists('./converted')==False:
            os.mkdir('./converted')
        for files in lstFilesDCM:
            dataset = pydicom.dcmread(files)
            Image.fromarray(dataset.pixel_array).save("./converted/img_" + str(saveNumber) + ".jpg")
            originalIMG = cv2.imread("./converted/img_" + str(saveNumber) + ".jpg")
            resizedIMG = self.image_resize(originalIMG, width=240, height=240)
            cv2.imwrite("./converted/img_" + str(saveNumber) + ".jpg", resizedIMG)
            saveNumber = saveNumber + 1

        print('Dataset transformed to JPG images into ./converted folder')