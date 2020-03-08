"Modulos propios"
import readDCM
"Modulos externos"
import os

def menu():

    os.system('cls')  # cls en windows, clear en linux
    print("Work modes:")
    print("\t1 - Read Single DCM image")
    print("\t2 - Still in development")
    print("\t3 - Still in development")
    print("\t9 - Exit")


while True:

    menu() #Muestra menu

    option = input("Select an option >> ")

    if option == "1":
        print("")

        found = 1
        while (found != 0):

            imgPath = input('Insert IMG name ->')

            if os.path.isfile(imgPath):

                print('DCM file found')
                obj = readDCM.readIMG()
                obj.readDCMimage(imgPath)
                found = 0

            else:

                print('DCM file not found')
                found = int(input('Insert 0 to exit, insert any number to try again'))
                if found == 0:
                    break


    elif option == "2":
        print("")
        input("Press any key to continue")
    elif option == "3":
        print("")
        input("Press any key to continue")
    elif option == "9":
        break
    else:
        print("")
        input("Wrong input, press any key to try again.")