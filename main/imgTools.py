import auxfunctions
import os
import cv2
import imutils

class imgTools:


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

    def resizeJPGfolder(self,path,width,height,destination):

        imgs=os.listdir(path)

        if os.path.exists(destination)==False:
            os.mkdir(destination)

        for i in imgs:
            print(i)
            original=cv2.imread(path+'/'+i)
            name,ext=auxfunctions.splitfilename(i)
            resized=self.image_resize(original, width=width, height=height)
            cv2.imwrite(destination+'/'+name+'.jpg', resized)


    def rotate(self,img,angle):

        original=cv2.imread(img)
        rotated=imutils.rotate(original,angle)

        return rotated