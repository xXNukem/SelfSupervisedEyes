from PIL import Image
import numpy as np
import random

img=Image.open('./converted/img_0.jpg')
#tamaño del cuadrado en pixels
squareSize=20
#Porcentaje de pixels que se pueden desplazar como maximo
squareSizePercent=50
#Pixeles que se pueden desplazar como maximo
movementPixels=(squareSize*squareSizePercent)/100

#Asertos-------------------
assert 3*squareSize<=img.width*img.height
assert squareSizePercent>=1 or squareSizePercent<=100

#--------------------------

#Obtencion recorte central aleatorio----------------------
#Generando cuadrado dentral aleatorio

#Genero un numero aleatorio que se le sumara o restara a las coordenadas centrales según una orientacion aleatoria
randX=random.uniform(1,movementPixels)
randY=random.uniform(1,movementPixels)

randomOrientation=random.randint(0,8)#Genero aleatoriamente hacia donde quiero que vaya más o menos el recorte central
print('Orientation:', randomOrientation)

#El recorte central se desplaza hacia la izquierda
if randomOrientation==1:
    xCenter = (img.width / 2) - randX
    yCenter = (img.height / 2)
    print('X:', xCenter, 'Y:', yCenter)
    xDist = (img.width / 2) - randX #xDist e yDist serán las varaibles para luego calcular los nuevos centros, no deben sobreescribirse con xCenter e yCenter
    yDist = (img.height / 2)

#El recorte central se desplaza hacia la diagonal superior izquierda
if randomOrientation==2:
    xCenter = (img.width / 2) - randX
    yCenter = (img.height / 2) - randY
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2) - randX
    yDist=(img.height / 2) - randY

#El recorte central se deplaza hacia arriba
if randomOrientation==3:
    xCenter = (img.width / 2)
    yCenter = (img.height / 2) - randY
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2)
    yDist=(img.height / 2) - randY

#El recorte central se desplaza a la diagonal superior derecha
if randomOrientation==4:
    xCenter = (img.width / 2) + randX
    yCenter = (img.height / 2) - randY
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2) + randX
    yDist=(img.height / 2) - randY

#El recorte central se desplaza a la derecha
if randomOrientation==5:
    xCenter = (img.width / 2) + randX
    yCenter = (img.height / 2) - randY
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2) + randX
    yDist=(img.height / 2) - randY

#El recorte central se desplaza a la exquina inferior derecha
if randomOrientation==6:
    xCenter = (img.width / 2) + randX
    yCenter = (img.height / 2) + randY
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2) + randX
    yDist=(img.height / 2) + randY

#El recorte central se desplaza hacia abajo
if randomOrientation==7:
    xCenter = (img.width / 2)
    yCenter = (img.height / 2) + randY
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2)
    yDist=(img.height / 2) + randY

#El recorte central se desplaza a la diagonal inferior izquierda
if randomOrientation==8:
    xCenter = (img.width / 2) - randX
    yCenter = (img.height / 2) + randY
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2) - randX
    yDist=(img.height / 2) + randY

#El recorte central se queda centrado
if randomOrientation==0:
    xCenter = (img.width / 2)
    yCenter = (img.height / 2)
    print('X:', xCenter, 'Y:', yCenter)
    xDist=(img.width / 2)
    yDist=(img.height / 2)

#Cortamos y guardamos el cuadrado central

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

centralSquareCropped=img.crop(croppingMask)

#centralSquareCropped.show('0')
centralSquareCropped.save('0.jpg')

#OBTENER EL RESTO DE RECORTES---------------------------------------------------------------------------------------------------

#Obtención del recorte 1 (izquierda)-----------------------------------------
#Distancia despecto del cuadradito central
centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2) #Asi se aleja o acerca más del recorte del centro
#Coordenadas respecto del central
xCenter=xDist-centralSquareDistance
yCenter=yDist+float(random.uniform(-movementPixels,movementPixels))

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
centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2)
#Coordenadas respecto del central
xCenter=xDist+float(random.uniform(-movementPixels,movementPixels))
yCenter=yDist-centralSquareDistance

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
centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2)
#Coordenadas respecto del central
xCenter=xDist+float(random.uniform(-movementPixels,movementPixels))
yCenter=yDist+centralSquareDistance

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
centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2) #Asi se aleja o acerca más del recorte del centro
#Coordenadas respecto del central
xCenter=xDist+centralSquareDistance
yCenter=yDist+float(random.uniform(-movementPixels,movementPixels))

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
centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2)
#Coordenadas respecto del central
xCenter=xDist-centralSquareDistance+float(random.uniform(-movementPixels,movementPixels))
yCenter=yDist-centralSquareDistance+float(random.uniform(-movementPixels,movementPixels))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

leftUpSquareCropped=img.crop(croppingMask)

leftUpSquareCropped.save('2.jpg')

#Superior derecha (4)
#Coordenadas respecto del cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2)
#Coordenadas respecto del central
xCenter=xDist+centralSquareDistance+float(random.uniform(-movementPixels,movementPixels))
yCenter=yDist-centralSquareDistance+float(random.uniform(-movementPixels,movementPixels))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

rightUpSquareCropped=img.crop(croppingMask)

rightUpSquareCropped.save('4.jpg')

#Inferior izquierda (8)

centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2)
#Coordenadas respecto del central
xCenter=xDist-centralSquareDistance-float(random.uniform(-movementPixels,movementPixels))
yCenter=yDist+centralSquareDistance+float(random.uniform(-movementPixels,movementPixels))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

leftDownSquareCropped=img.crop(croppingMask)

leftDownSquareCropped.save('8.jpg')

#Inferior derecha(6)
#Coordenadas respecto del cuadrado central
centralSquareDistance=(squareSize/2)+float((random.randrange(movementPixels)+movementPixels))+(squareSize/2)
#Coordenadas respecto del central
xCenter=xDist+centralSquareDistance+float(random.uniform(-movementPixels,movementPixels))
yCenter=yDist+centralSquareDistance+float(random.uniform(-movementPixels,movementPixels))

x1=xCenter-(squareSize/2)
y1=yCenter-(squareSize/2)
x2=xCenter+(squareSize/2)
y2=yCenter+(squareSize/2)

croppingMask=(x1,y1,x2,y2)

rightDownSquareCropped=img.crop(croppingMask)

rightDownSquareCropped.save('6.jpg')