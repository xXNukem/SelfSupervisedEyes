import sys
import os
sys.path.append("..")
import importlib.util

#-------------------------------

#main menu for dataset creation
def menu():

    os.system('cls') #change to 'clear' for linux
    print("Work modes:")
    print('General -------------------------------------------')
    print('\t1 - Resize JPEG img folder')
    print('\t2 - Categorize Dataset')
    print('\t3 - Undersample category')
    print('\t4 - Oversample category')
    print('\t5 - Preprocess img directory')
    print('Context Prediction Method -------------------------')
    print("\t6 - Generate Dataset")
    print('\t7 - Generate train/validation split for Context Prediction')
    print('Rotation Method------------------------------------')
    print('\t8 - Generate Dataset')
    print('Jiggsaw Permutation method-------------------------')
    print('\t9 - Generate Dataset')
    print('\t10 - Generate train/validation split for JiggSaw')
    print("11 - Exit")


while True:
    #import and calling other python modules
    import imgTools
    import dataBalancing
    import preprocessing

    os.system('cls')
    spec = importlib.util.spec_from_file_location("contexPredictionFunctions.py",
                                                  "../contextPrediction/"
                                                  "contextPredictionFunctions.py")
    contextPrediction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contextPrediction)
    spec = importlib.util.spec_from_file_location("rotationFunctions.py",
                                                  "../rotation/"
                                                  "rotationFunctions.py")
    rotation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rotation)
    spec = importlib.util.spec_from_file_location("jiggsawFunctions.py",
                                                  "../jiggsaw/"
                                                  "jiggsawFunctions.py")
    jiggsaw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jiggsaw)

    menu()

    option = input("Select an option >> ")

    imgTools = imgTools.imgTools()
    contextPrediction=contextPrediction.contextPrediction()
    rotation=rotation.rotation()
    jiggsaw=jiggsaw.jiggsaw()
    dataBalancing=dataBalancing.dataBalancing()
    preprocessing=preprocessing.preprocessing()

    if option == '1':
        print('This will resize an entire folder filled with images')
        imgPath=input('Enter IMG path -> ')
        width=int(input('Enter width -> '))
        height=int(input('Enter height -> '))
        r=input('Preserve aspect ratio? 1-Yes Any key-No -> ')
        destination=input('Enter the path of the destination folder -> ')
        imgTools.resizeJPGfolder(imgPath,width,height,destination,r)
        input("Press any key to continue")

    elif option=='2':
        print('This will categorize your dataset in folders with the help of a .csv file')
        imgPath = input('Enter the path of your images ->')
        destPath = input('Enter the path for the categorized dataset ->')
        csv = input('Enter the path of the .csv file ->')
        nCategories = int(input('Enther the number of categories ->'))
        dataBalancing.categorizeDataset(csv,imgPath,destPath,nCategories)
        input("Press any key to continue")

    elif option == "3":
        print("This will undersample a category until the number of samples specified")
        path = input('Enter the category path to undersample ->')
        nCategories= int(input('Enter the number of categories to undersample ->'))
        dataBalancing.undersampleCategory(path,nCategories)
        input("Press any key to continue")

    elif option=='4':
        print("This will oversample a category until the number of samples specified with data augmentation")
        path = input('Enter the category path to ondersample ->')
        nCategories = int(input('Enter the number of categories to undersample ->'))
        dataBalancing.oversampleCategory(path, nCategories)
        input("Press any key to continue")

    elif option=='5':
        print("This will preprocess a directory with imgs inside")
        path = input('Enter the path to preprocess ->')
        destPath =input('Enter the path for the preprocessed imgs ->')
        contrastFilter = int(input('Enter the minimum STD for the low contrast filter '
                                   '(0 for no low contrast filter) ->'))
        nCores = int(input('Enter the number of cores of your CPU to increase preprocessing speed ->'))
        preprocessing.launchPreprocessing(path,destPath,nCores,contrastFilter)
        input("Press any key to continue")

    elif option == "6":
        print("This will generate the Dataset")
        imgPath = input('Enter IMG Path -> ')
        sqSize = int(input('Enter the square size -> '))
        sqPercent = int(input('Enter the percentaje of variation in the square distances -> '))
        pathname = input('Enter the destination path -> ')
        contextPrediction.generateDataset(imgPath, sqSize, sqPercent, pathname)
        input("Press any key to continue")

    elif option == '7':
        print('This will split your dataset into train and validation sets and save into .pickle files')
        datasetPath = input('Enter the path of your dataset -> ')
        percent = int(input('Enter percentaje for validation -> '))
        imglist = contextPrediction.loadimgspath(datasetPath)
        contextPrediction.splitGenerator(imglist, percent)
        input("Press any key to continue")

    elif option=='8':
        print('This will generate the dataset')
        imgPath=input('Enter IMG path -> ')
        pathname = input('Enter the destination path -> ')
        rotation.generateDataset(imgPath,pathname)
        input("Press any key to continue")


    elif option == "9":
        print("This will generate the Dataset")
        imgPath=input('Enter IMG Path -> ')
        sqSize=int(input('Enter the square size -> '))
        sqPercent = int(input('Enter the percentaje of variation in the square distances -> '))
        pathname=input('Enter a name for the folder wich will contain the dataset -> ')
        jiggsaw.generateDataset(imgPath,sqSize,pathname,sqPercent)
        input("Press any key to continue")

    elif option=='10':
        print('This will split your dataset into train and validation sets and save into .pickle files')
        datasetPath=input('Enter the path of your dataset -> ')
        maxDist=int(input('Enter the maximum Hamming Distance (equal to nº Classes) -> '))
        percent = int(input('Enter percentaje for validation -> '))
        imglist=jiggsaw.loadimgspath(datasetPath,maxDist)
        jiggsaw.splitGenerator(imglist,percent)
        input("Press any key to continue")


    elif option == "11":
        break
    else:
        print("")
        input("Wrong input, press any key to try again.")