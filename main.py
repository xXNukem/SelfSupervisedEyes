"Modulos propios"
import readDCM
"Modulos externos"
import os

def menu():

    os.system('cls')  # cls en windows, clear en linux
    print("Work modes:")
    print("\t1 - Read single DCM image")
    print("\t2 - Transform DCM dataset into JPG images")
    print("\t3 - Generate Dataset")
    print("\t9 - Exit")


while True:

    menu()

    option = input("Select an option >> ")

    obj = readDCM.readIMG()

    if option == "1":
        print("This will show all the info of a DCM image given ")
        found = 1
        while (found != 0):

            imgPath = input('Insert IMG name -> ')

            if os.path.isfile(imgPath):

                print('DCM file found')
                obj.readDCMimage(imgPath)
                found = 0

            else:

                print('DCM file not found')
                found = int(input('Insert 0 to exit, insert any number to try again'))
                if found == 0:
                    break


    elif option == "2":
        print("This will read a DCM dataset and transform it to JPG images with a new resolution")
        datasetPath=input('Enter dataset path ->')
        width=int(input('Width - >'))
        height=int(input('Height ->'))
        obj.readDCMdataset(datasetPath,width,height)

    elif option == "3":
        print("This will generate the Dataset")
        imgPath=input('Enter IMG Path -> ')
        sqSize=int(input('Enter the square size -> '))
        sqPercent=int(input('Enter the percentaje of variation in the distances -> '))
        obj.generateDataset(imgPath,sqSize,sqPercent)
        input("Press any key to continue")

    elif option == "9":
        break
    else:
        print("")
        input("Wrong input, press any key to try again.")