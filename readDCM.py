import matplotlib.pyplot as plt
import pydicom
import os
from PIL import Image, ImageOps
import numpy as np

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
                originalIMG = Image.open("./converted/img_" + str(saveNumber) + ".jpg")
                resizedIMG = originalIMG.resize((240, 240))
                resizedIMG.save("./converted/img_" + str(saveNumber) + ".jpg")
                saveNumber = saveNumber + 1
        else:
            for files in lstFilesDCM:
                dataset=pydicom.dcmread(files)
                Image.fromarray(dataset.pixel_array).save("./converted/img_"+str(saveNumber)+".jpg")
                originalIMG=Image.open("./converted/img_"+str(saveNumber)+".jpg")
                resizedIMG=originalIMG.resize((240,240))
                resizedIMG.save("./converted/img_"+str(saveNumber)+".jpg")
                saveNumber=saveNumber+1

        print('Dataset transformed to JPG images into ./converted folder')