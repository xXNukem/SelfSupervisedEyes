import matplotlib.pyplot as plt
import pydicom

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
