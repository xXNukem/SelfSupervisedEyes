"Modulos propios"
import readDCM
import auxfunctions
"Modulos externos"
import os

def menu():

    os.system('cls')  # cls en windows, clear en linux
    print("Work modes:")
    print("\t1 - Read single DCM image")
    print("\t2 - Transform DCM dataset into JPG images")
    print("\t3 - Generate Dataset")
    print('\t4 - Resize JPG images')
    print('\t5 - Generate train/validation split')
    print('\t6 - Calculate mean and STD')
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
        pathname=input('Enter a name for the folder wich will contain the dataset -> ')
        obj.generateDataset(imgPath,sqSize,sqPercent,pathname)
        input("Press any key to continue")

    elif option == '4':
        print('This will resize a folder wich contains JPG images')
        imgPath=input('Enter IMG path -> ')
        width=int(input('Enter width -> '))
        height=int(input('Enter height -> '))
        destination=input('Enter the name of the destination folder -> ')
        obj.resizeJPGfolder(imgPath,width,height,destination)
        input("Press any key to continue")

    elif option=='5':
        print('This will split your dataset into train and validation sets and save into .pickle files')
        datasetPath=input('Enter the path of your dataset ->')
        percent=int(input('Enter percentaje for validation -> '))
        imglist=auxfunctions.loadimgspath(datasetPath)
        auxfunctions.splitGenerator(imglist,percent)
        input("Press any key to continue")

    elif option=='6':
        print('This will calculate mean and STD of a dataset and save into .pickle files')
        datasetPath = input('Enter the path of your dataset ->')
        auxfunctions.calculateMeanStd(datasetPath)
        input("Press any key to continue")

    elif option == "9":
        break
    else:
        print("")
        input("Wrong input, press any key to try again.")