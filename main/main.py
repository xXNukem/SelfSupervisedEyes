import sys
import imgTools
import os
sys.path.append("..")
import importlib.util
spec = importlib.util.spec_from_file_location("contexPredictionFunctions.py", "../contextPrediction/contextPredictionFunctions.py")
contextPrediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contextPrediction)
spec = importlib.util.spec_from_file_location("contexPredictionFunctions.py", "../classification/classificationFunctions.py")
classification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classification)
spec = importlib.util.spec_from_file_location("rotationFunctions.py", "../rotation/rotationFunctions.py")
rotation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotation)
spec = importlib.util.spec_from_file_location("jiggsawFunctions.py", "../jiggsaw/jiggsawFunctions.py")
jiggsaw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jiggsaw)
#------------------------------------------------------------------------------------------------


def menu():

    os.system('cls')  # cls en windows, clear en linux
    print("Work modes:")
    print('General -------------------------------------------')
    print('\t1 - Resize JPG img folder')
    print('\t2 - Calculate mean and STD for Classification')
    print('Context Prediction Method -------------------------')
    print("\t3 - Generate Dataset")
    print('\t4 - Generate train/validation split for Context Prediction')
    print('\t5 - Calculate mean and STD for Context Prediction')
    print('Rotation Method------------------------------------')
    print('\t6 - Generate Dataset')
    print('\t7 - Calculate mean and STD for Rotation')
    print('Jiggsaw Permutation method-------------------------')
    print('\t8 - Generate Dataset')
    print('\t9 - Generate train/validation split for JiggSaw')
    print('\t10 - Calculate mean and STD for Context Prediction')
    print("11 - Exit")


while True:

    menu()

    option = input("Select an option >> ")

    imgTools = imgTools.imgTools()
    contextPrediction=contextPrediction.contextPrediction()
    rotation=rotation.rotation()
    jiggsaw=jiggsaw.jiggsaw()
    classification=classification.classification()

    if option == '1':
        print('This will resize a folder wich contains JPG images')
        imgPath=input('Enter IMG path -> ')
        width=int(input('Enter width -> '))
        height=int(input('Enter height -> '))
        r=input('Preserve aspect ratio? 1-Yes Any key-No -> ')
        destination=input('Enter the name of the destination folder -> ')
        imgTools.resizeJPGfolder(imgPath,width,height,destination,r)
        input("Press any key to continue")

    elif option=='2':
        print('This will calculate mean and STD of a dataset and save into .pickle files')
        datasetPath = input('Enter the path of your dataset ->')
        classification.calculateMeanStdClassification(datasetPath)
        input("Press any key to continue")


    elif option == "3":
        print("This will generate the Dataset")
        imgPath=input('Enter IMG Path -> ')
        sqSize=int(input('Enter the square size -> '))
        sqPercent=int(input('Enter the percentaje of variation in the distances -> '))
        pathname=input('Enter a name for the folder wich will contain the dataset -> ')
        contextPrediction.generateDataset(imgPath,sqSize,sqPercent,pathname)
        input("Press any key to continue")


    elif option=='4':
        print('This will split your dataset into train and validation sets and save into .pickle files')
        datasetPath=input('Enter the path of your dataset -> ')
        percent=int(input('Enter percentaje for validation -> '))
        imglist=contextPrediction.loadimgspath(datasetPath)
        contextPrediction.splitGenerator(imglist,percent)
        input("Press any key to continue")

    elif option=='5':
        print('This will calculate mean and STD of a dataset and save into .pickle files')
        datasetPath = input('Enter the path of your dataset -> ')
        contextPrediction.calculateMeanStd(datasetPath)
        input("Press any key to continue")

    elif option=='6':
        print('This will generate the dataset')
        imgPath=input('Enter IMG path -> ')
        pathname = input('Enter a name for the folder wich will contain the dataset -> ')
        rotation.generateDataset(imgPath,pathname)
        input("Press any key to continue")

    elif option=='7':
        print('This will calculate mean and STD of a dataset for Rotation Method and save into .pickle files')
        datasetPath = input('Enter the path of your dataset ->')
        rotation.calculateMeanStd(datasetPath)
        input("Press any key to continue")

    elif option == "8":
        print("This will generate the Dataset")
        imgPath=input('Enter IMG Path -> ')
        sqSize=int(input('Enter the square size -> '))
        pathname=input('Enter a name for the folder wich will contain the dataset -> ')
        jiggsaw.generateDataset(imgPath,sqSize,pathname)
        input("Press any key to continue")

    elif option=='9':
        print('This will split your dataset into train and validation sets and save into .pickle files')
        datasetPath=input('Enter the path of your dataset -> ')
        maxDist=int(input('Enter the maximum Hamming Distance (equal to nº Classes) -> '))
        percent = int(input('Enter percentaje for validation -> '))
        imglist=jiggsaw.loadimgspath(datasetPath,maxDist)
        jiggsaw.splitGenerator(imglist,percent)
        input("Press any key to continue")

    elif option=='10':
        print('This will calculate mean and STD of a dataset for Rotation Method and save into .pickle files')
        datasetPath = input('Enter the path of your dataset ->')
        jiggsaw.calculateMeanStd(datasetPath)
        input("Press any key to continue")


    elif option == "11":
        break
    else:
        print("")
        input("Wrong input, press any key to try again.")