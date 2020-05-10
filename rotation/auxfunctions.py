#Obtiene el nombre y la extension de un archivo
def splitfilename(filename):
    sname=""
    sext=""
    i=filename.rfind(".")
    if(i!=0):
        n=len(filename)
        j=n-i-1
        sname=filename[0:i]
        sext=filename[-j:]
    return sname, sext

def splitGetAngle(filename):
    splitter='_'
    num,name,angle = filename.split(splitter)
    splitter2='.'
    angle,ext=angle.split(splitter2)

    return angle