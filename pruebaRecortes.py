from PIL import Image
import numpy as np
import random

img=Image.open('./converted/img_0.jpg')
#izquierda,arriba,derecha,abajo
#240*240

randomXCenter=random.uniform(1.5,3.5)
randomYCenter=random.uniform(1.5,3.5)

xCenter=img.width/randomXCenter
yCenter=img.height/randomYCenter
x1=xCenter-20
y1=yCenter-20
x2=xCenter+20
y2=yCenter+20

croppingMask=(x1,y1,x2,y2)

central_square_cropped=img.crop(croppingMask)

print(central_square_cropped.width)
print(central_square_cropped.height)

central_square_cropped.show()