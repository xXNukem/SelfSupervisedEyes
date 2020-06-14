import cv2, os
import numpy as np
import matplotlib as plt
from joblib import Parallel, delayed

#Code adapted from https://www.kaggle.com/orkatz2/diabetic-retinopathy-preprocess-rgba-update
class preprocessing:

    #get retina radius from an img
    def estimate_radius(self,img):
        mx = img[img.shape[0] // 2, :, :].sum(1)
        rx = (mx > mx.mean() / 10).sum() / 2

        my = img[:, img.shape[1] // 2, :].sum(1)
        ry = (my > my.mean() / 10).sum() / 2

        return (ry, rx)

    #subtract gaussian blur filter from an
    def subtract_gaussian_blur(self,img):

        gb_img = cv2.GaussianBlur(img, (0, 0), 5) #get filter

        return cv2.addWeighted(img, 4, gb_img, -4, 128) #return img-filter

    #remove outer circle
    def remove_outer_circle(self,a, p, r):
        b = np.zeros(a.shape, dtype=np.uint8)
        cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)

        return a * b + 128 * (1 - b)

    #crop black background
    def crop_img(self,img, h, w):
        h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
        w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0

        crop_img = img[h_margin:h + h_margin, w_margin:w + w_margin, :]

        return crop_img

    #put img in the corresponding square after removing outer circle
    def place_in_square(self,img, r, h, w):
        new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
        new_img += 128
        new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img

        return new_img

    #launch all preprocess functions
    #f: img file
    #r: estimated radius (512 to preserve original image resolution)
    def preprocess(self,f, r, debug_plot=False):
        try:
            img = cv2.imread(f)

            ry, rx = self.estimate_radius(img)

            if debug_plot:
                plt.figure()
                plt.imshow(img)

            resize_scale = r / max(rx, ry)
            w = min(int(rx * resize_scale * 2), r * 2)
            h = min(int(ry * resize_scale * 2), r * 2)

            img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)

            img = self.crop_img(img, h, w)
            print("crop_img", np.mean(img), np.std(img))

            if debug_plot:
                plt.figure()
                plt.imshow(img)

            img = self.subtract_gaussian_blur(img)
            img = self.remove_outer_circle(img, 0.9, r)
            img = self.place_in_square(img, r, h, w)

            if debug_plot:
                plt.figure()
                plt.imshow(img)

            return img

        except Exception as e:
            print("file {} exception {}".format(f, e))

        return None

    #save the preprocessed imgs in destPath
    #f: path of an unpreprocessed img
    #destPath: Path for the preprocessed imgs
    #f2: name of the file
    #file: Low contrast filter value
    def process_and_save(self,f,destPath,f2,filter):

        print("processing:", f)

        result = self.preprocess(f, 512) #512 radius to preserver original img resolution

        if result is None:
            return

        #Filter low contrast images

        if filter > 0:
            std = np.std(result)
            if std < filter:
                print("skip low std", std, f)
                return

        if result is not None:
            print(cv2.imwrite(destPath+'/'+f2, result))

    #launch preprocessing operation
    #path: Path for the original imgs without preprocessing
    #destPath: Path for the preprocessed imgs
    #nCores: Number of CPU cores to increase preprocessing speed
    #filter: Value of the low contrast filter, 0 will not apply this filter
    def launchPreprocessing(self,path,destPath,nCores,filter):

        if os.path.exists(destPath) == False:
            os.mkdir(destPath)

        train_files=os.listdir(path)


        Parallel(n_jobs=nCores)(delayed(self.process_and_save)
                                (path+'/'+f,destPath,f,filter) for f in train_files)
