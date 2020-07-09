import click
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageDraw, ImageFont
import matplotlib as plt
import os
import cv2

#params
@click.command()
@click.option('--test_img', '-t', default=None, required=True, help=u'Validation directory.')
@click.option('--model_files', '-M', default=None, required=True, help=u'Model file.')
@click.option('--classification', '-c', is_flag=True,
              help=u' -c for a classification model, nothing for a regression model.')
@click.option('--test_dir', '-d', is_flag=True,
              help=u' -d to enter a directory with images (using -t argument)')

def getPrediction(test_img,model_files,classification,test_dir):

    #--------------PREPROCESSING FUNCTIONS---------------------------------------------
    # get retina radius from an img
    def estimate_radius( img):
        mx = img[img.shape[0] // 2, :, :].sum(1)
        rx = (mx > mx.mean() / 10).sum() / 2

        my = img[:, img.shape[1] // 2, :].sum(1)
        ry = (my > my.mean() / 10).sum() / 2

        return (ry, rx)

    # subtract gaussian blur filter from an
    def subtract_gaussian_blur(img):

        gb_img = cv2.GaussianBlur(img, (0, 0), 5)  # get filter

        return cv2.addWeighted(img, 4, gb_img, -4, 128)  # return img-filter

    # remove outer circle
    def remove_outer_circle(a, p, r):
        b = np.zeros(a.shape, dtype=np.uint8)
        cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)

        return a * b + 128 * (1 - b)

    # crop black background
    def crop_img( img, h, w):
        h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
        w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0

        crop_img = img[h_margin:h + h_margin, w_margin:w + w_margin, :]

        return crop_img

    # put img in the corresponding square after removing outer circle
    def place_in_square( img, r, h, w):
        new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
        new_img += 128
        new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img

        return new_img

    # launch all preprocess functions
    # f: img file
    # r: estimated radius (512 to preserve original image resolution)
    def preprocess( f, r, debug_plot=False):
        try:
            f.encode('utf-8')
            img = cv2.imread(f)

            ry, rx = estimate_radius(img)

            if debug_plot:
                plt.figure()
                plt.imshow(img)

            resize_scale = r / max(rx, ry)
            w = min(int(rx * resize_scale * 2), r * 2)
            h = min(int(ry * resize_scale * 2), r * 2)

            img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)

            img = crop_img(img, h, w)
            #print("crop_img", np.mean(img), np.std(img))

            if debug_plot:
                plt.figure()
                plt.imshow(img)

            img = subtract_gaussian_blur(img)
            img = remove_outer_circle(img, 0.9, r)
            img = place_in_square(img, r, h, w)

            if debug_plot:
                plt.figure()
                plt.imshow(img)

            return img

        except Exception as e:
            print("file {} exception {}".format(f, e))

        return None
#-----------------------------------------------------------------------------------------------------

    model =model_files
    cnn = tf.keras.models.load_model(model)

    if test_dir==False:

        original=test_img

        preprocessed=preprocess(test_img,512)
        cv2.imwrite('./temp.jpeg',preprocessed)
        x = load_img('./temp.jpeg', target_size=(224, 224)) ##VGG16 shape target
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        predictions = cnn.predict(x)

        if classification==True:
            result = predictions[0]
            print(predictions[0])
            answer = np.argmax(result)
            percent=predictions[0][answer]
        else:
            answer=round(float(predictions))
            print(predictions)

        image = Image.open(original) #instance to draw text into a image
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 40) #you must give the entire font path in unix for this to work

        if classification==True:
            if answer == 0:
                draw.text((50, 50), "No Diabetic Retinopathy"+' \t Score: '+"{:.2f}".format(percent*100)+'%', font=font, fill="white")
                image.show()
            elif answer == 1:
                draw.text((50, 50), "Mild"+' \t Score: '+"{:.2f}".format(percent*100)+'%', font=font, fill="white")
                image.show()
            elif answer == 2:
                draw.text((50, 50), "Moderate"+' \t Score: '+"{:.2f}".format(percent*100)+'%', font=font, fill="white")
                image.show()
            elif answer == 3:
                draw.text((50, 50), "Severe"+' \t Score: '+"{:.2f}".format(percent*100)+'%', font=font, fill="white")
                image.show()
            elif answer == 4:
                draw.text((50, 50), "Proliferative"+' \t Score: '+"{:.2f}".format(percent*100)+'%', font=font, fill="white")
                image.show()
        else:
            if answer == 0:
                draw.text((50, 50), "No Diabetic Retinopathy"+' \t Est Class: '+"{:.2f}".format(float(predictions)), font=font, fill="white")
                image.show()
            elif answer == 1:
                draw.text((50, 50), "Mild"+' \t Est Class: '+"{:.2f}".format(float(predictions)), font=font, fill="white")
                image.show()
            elif answer == 2:
                draw.text((50, 50), "Moderate"+' \t Est Class: '+"{:.2f}".format(float(predictions)), font=font, fill="white")
                image.show()
            elif answer == 3:
                draw.text((50, 50), "Severe"+' \t Est Class: '+"{:.2f}".format(float(predictions)), font=font, fill="white")
                image.show()
            elif answer == 4:
                draw.text((50, 50), "Proliferative"+' \t Est Class: '+"{:.2f}".format(float(predictions)), font=font, fill="white")
                image.show()

            os.remove('./temp.jpeg')
    else:
        imgs=os.listdir(test_img)

        for i in imgs:
            print('\n')
            preprocessed = preprocess(test_img+'/'+i, 512)
            cv2.imwrite('./temp.jpeg', preprocessed)

            x = load_img('./temp.jpeg', target_size=(224, 224))  ##VGG16 shape target
            x = img_to_array(x)
            x = np.expand_dims(x, axis=0)
            predictions = cnn.predict(x)

            if classification == True:
                result = predictions[0]
                #print(predictions[0])
                answer = np.argmax(result)
                #percent = predictions[0][answer]
                #print("{:.2f}".format(percent * 100))
            else:
                answer = round(float(predictions))
                print(predictions)

            print('Img name: ', test_img+'/'+i)

            if answer==0:
                print(predictions)
                print('Diagnose -> No Diabetic Retinopaty')
            elif answer==1:
                print(predictions)
                print('Diagnose -> Mild')
            elif answer==2:
                print(predictions)
                print('Diagnose -> Moderate')
            elif answer==3:
                print(predictions)
                print('Diagnose -> Severe')
            elif answer==4:
                print(predictions)
                print('Diagnose -> Proliferative')

            os.remove('./temp.jpeg')

getPrediction()