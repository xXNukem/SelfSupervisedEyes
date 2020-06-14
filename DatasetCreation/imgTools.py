import os
import cv2
import imutils

class imgTools:

    #resize imgs preserving aspect ratio
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

    #resize a folder filled with imgs
    def resizeJPGfolder(self,path,width,height,destination,r):

        imgs=os.listdir(path)

        if os.path.exists(destination)==False:
            os.mkdir(destination)

        if r=='1':
            for i in imgs:
                print(i)
                original=cv2.imread(path+'/'+i)
                name,ext=self.splitfilename(i)
                resized=self.image_resize(original, width=width, height=height)
                cv2.imwrite(destination+'/'+name+'.jpeg', resized)
        else:
            for i in imgs:
                print(i)
                original=cv2.imread(path+'/'+i)
                name,ext=self.splitfilename(i)
                resized = cv2.resize(original, (width,height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(destination + '/' + name + '.jpeg', resized)

    #rotate imgs
    def rotate(self,img,angle):

        original=cv2.imread(img)
        rotated=imutils.rotate(original,angle)

        return rotated

    #get name and extension from any file
    def splitfilename(self,filename):
        sname = ""
        sext = ""
        i = filename.rfind(".")
        if (i != 0):
            n = len(filename)
            j = n - i - 1
            sname = filename[0:i]
            sext = filename[-j:]
        return sname, sext