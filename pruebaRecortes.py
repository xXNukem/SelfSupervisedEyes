from PIL import Image
import numpy as np
import random

img=Image.open('./converted/img_0.jpg')
#izquierda,arriba,derecha,abajo
#240*240

#Obtención recorte central
randomCenter=random.uniform(1.5,3.5)
randomXCenter=randomCenter
randomYCenter=randomCenter
squareSize=70

xCenter=img.width/randomXCenter
yCenter=img.height/randomYCenter
x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

centralSquareCropped=img.crop(croppingMask)

#centralSquareCropped.show('0')
centralSquareCropped.save('0.jpg')


#Obtención del recorte 1 (izquierda)-----------------------------------------
#Distancia despecto del cuadradito central
centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2) #Asi se aleja o acerca más del recorte del centro
#Coordenadas respecto del central
xCenter=img.width/randomXCenter-centralSquareDistance
yCenter=img.height/randomYCenter+float(random.uniform(-10.5,10.5))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

leftSquareCropped=img.crop(croppingMask)

#leftSquareCropped.show('img1')
leftSquareCropped.save('1.jpg')


#Obtención del recorte 3 (arriba)-------------------------------------
#Distancia respecto al cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2)
#Coordenadas respecto del central
xCenter=img.width/randomXCenter+float(random.uniform(-10.5,10.5))
yCenter=img.height/randomXCenter-centralSquareDistance

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

upSquareCropped=img.crop(croppingMask)

#upSquareCropped.show('img3')
upSquareCropped.save('3.jpg')


#Obtencion del recorte 7 (abajo)
#Distancia respecto al cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2)
#Coordenadas respecto del central
xCenter=img.width/randomXCenter+float(random.uniform(-10.5,10.5))
yCenter=img.height/randomXCenter+centralSquareDistance

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

rightSquareCropped=img.crop(croppingMask)

#rightSquareCropped.show('img5')
rightSquareCropped.save('7.jpg')
"""

#Obtencion del recorte 7 (abajo)
#Distancia respecto al cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(9)+1))
#Coordenadas respecto del central
xCenter=img.width/randomXCenter-centralSquareDistance
yCenter=img.height/randomXCenter+centralSquareDistance+random.uniform(-5.5,5.5)

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

downSquareCropped=img.crop(croppingMask)

#downSquareCropped.show('img7')
downSquareCropped.save('7.jpg')
"""