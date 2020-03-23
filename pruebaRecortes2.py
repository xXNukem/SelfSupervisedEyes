from PIL import Image
import numpy as np
import random

img=Image.open('./converted/img_0.jpg')
#izquierda,arriba,derecha,abajo
#240*240

#Obtención recorte central
#randomCenter=random.uniform(1.5,3.5)
#randomXCenter=random.uniform(1.5,2.5)
#randomYCenter=random.uniform(1.5,2.5)
squareSize=40

xCenter=(img.width/2)+random.uniform(-30,30)
yCenter=(img.height/2)+random.uniform(-30,30)
print('X:',xCenter,'Y:',yCenter)
x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

centralSquareCropped=img.crop(croppingMask)

#centralSquareCropped.show('0')
centralSquareCropped.save('0.jpg')
"""
randomXCenter=xCenter
randomYCenter=yCenter
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
yCenter=img.height/randomYCenter-centralSquareDistance

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
yCenter=img.height/randomYCenter+centralSquareDistance

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

downSquareCropped=img.crop(croppingMask)

#rightSquareCropped.show('img5')
downSquareCropped.save('7.jpg')


#Obtencion del recorte 5 (derecha)
#Distancia despecto del cuadradito central
centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2) #Asi se aleja o acerca más del recorte del centro
#Coordenadas respecto del central
xCenter=img.width/randomXCenter+centralSquareDistance
yCenter=img.height/randomYCenter+float(random.uniform(-10.5,10.5))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

rightSquareCropped=img.crop(croppingMask)

#downSquareCropped.show('img7')
rightSquareCropped.save('5.jpg')

#OBTENCION DE LAS DIAGONALES----------------------------------------------------------------------------------

#Superior izquierda (2)
#Coordenadas respecto del cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2)
#Coordenadas respecto del central
xCenter=img.width/randomXCenter-centralSquareDistance+float(random.uniform(-10.5,10.5))
yCenter=img.height/randomYCenter-centralSquareDistance+float(random.uniform(-10.5,10.5))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

leftUpSquareCropped=img.crop(croppingMask)

leftUpSquareCropped.save('2.jpg')

#Superior derecha (4)
#Coordenadas respecto del cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2)
#Coordenadas respecto del central
xCenter=img.width/randomXCenter+centralSquareDistance+float(random.uniform(-10.5,10.5))
yCenter=img.height/randomYCenter-centralSquareDistance+float(random.uniform(-10.5,10.5))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

rightUpSquareCropped=img.crop(croppingMask)

rightUpSquareCropped.save('4.jpg')

#Inferior izquierda (8)

centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2)
#Coordenadas respecto del central
xCenter=img.width/randomXCenter-centralSquareDistance+float(random.uniform(-10.5,10.5))
yCenter=img.height/randomYCenter+centralSquareDistance+float(random.uniform(-10.5,10.5))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

leftDownSquareCropped=img.crop(croppingMask)

leftDownSquareCropped.save('8.jpg')

#Inferior derecha(6)
#Coordenadas respecto del cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(20)+10))+(squareSize/2)
#Coordenadas respecto del central
xCenter=img.width/randomXCenter+centralSquareDistance+float(random.uniform(-10.5,10.5))
yCenter=img.height/randomYCenter+centralSquareDistance+float(random.uniform(-10.5,10.5))

croppingMask=(x1,y1,x2,y2)

rightDownSquareCropped=img.crop(croppingMask)

rightDownSquareCropped.save('6.jpg')
"""