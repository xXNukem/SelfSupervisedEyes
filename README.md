# Comparative study of self-supervised approaches applied to ocular images

### Description
This works pretends to apply Self Supervised training to improve diabetic retinopathy level prediction.

![image](C:\Users\cbp_1\Desktop\Universidad\TFG\cpexampleprep.png)

### Usage
First, you have to prepare the data for training each Self-Supervised approach, so go to the terminal and launch:

```sh
$ python ./datasetCreation/main.py
```
Now you will see the main menu where you can access all the options for data generation.
It is recommended to follow the next steps:
 - Preprocess the entire set of images
 - Resize the entire set of images
 - Prepare the data for each Self-Supervised training:
     - For Context Prediction and Jiggsaw you will need to make aditional train/validation splits, those files will be saved automatically in ContextPrediction folder and Jiggsaw Folder and will be read automatically when you launch those trainings.
     - This step is not required for the Rotation method.

Now you can train each Self-Supervised method, so you can launch those commands on the terminal:

 * Context Prediction:
```sh
$ python ./contextPrediction/contextPrediction.py 
```
 * Jiggsaw:
```sh
$ python ./jiggsaw/jiggsaw.py
```
 * Rotation:
```sh
$ python ./rotation/rotation.py
```
Remember that those executions need arguments to work, so you can use ``` --help ``` to get a list with all the arguments. Most of them are set by default, so you will have to worry about the img size and train/validation splits in case of Rotation.

Now you can import the weights that you got in Self-Supervised to train a model for diabetic retinopathy level prediction. Those weights will be automatically saved in the same Self-Supervised folder after training. (Before this, remember to go to the main menu and select 'Categorize Dataset' option to categorize diabetic retinopathy images in folders).
So you can go to the terminal and launch this:

```sh
$ python ./trainTest/trainVGG16.py -l -w 'path to the weights'
```
Remember, you can use ``` --help ``` command to see more arguments.

Also each training generates a training history and a model file, so you can use that model file to watch the metrics executing:
```sh
$ python ./trainTest/getModelReport.py -V 'path to test split' -M 'path to model file'
```
Also you can enter ``` --help ``` to see all the arguments that you can use.
