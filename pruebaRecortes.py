from PIL import Image
import numpy as np
import random

img=Image.open('./converted/img_0.jpg')
#izquierda,arriba,derecha,abajo
#240*240

#Obtención recorte central
randomXCenter=random.uniform(1.5,3.5)
randomYCenter=random.uniform(1.5,3.5)
squareSize=40

xCenter=img.width/randomXCenter
yCenter=img.height/randomYCenter
x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

centralSquareCropped=img.crop(croppingMask)

centralSquareCropped.show('0')

#Obtención del recorte 1 (izquierda)-----------------------------------------
#Distancia despecto del cuadradito central
centralSquareDistance=(squareSize/2)+float((random.randrange(9)+1)) #Asi se aleja o acerca más del recorte del centro
#Coordenadas respecto del central
xCenter=img.width/randomXCenter-centralSquareDistance+random.uniform(-5.5,5.5) #Así muevo el cuadro un poco arriba o abajo
yCenter=img.height/randomYCenter-centralSquareDistance

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

centralSquareCropped=img.crop(croppingMask)

centralSquareCropped.show('1')