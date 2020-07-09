import click
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error
import numpy as np
import tensorflow as tf
#params
@click.command()
@click.option('--validation_files', '-V', default=None, required=True, help=u'Validation directory.')
@click.option('--model_files', '-M', default=None, required=True, help=u'Model file.')
@click.option('--classification', '-c', is_flag=True,
              help=u' -c for a classification model, nothing for a regression model.')

#get model metrics
def testModel(validation_files,classification,model_files):
    model = model_files
    #model_weights = './VGG16_ROTATION_REGRESSION(HUBERLOSS)_weights.h5'
    model = tf.keras.models.load_model(model)

    evaluation_generator=ImageDataGenerator(validation_split=0.99)


    evaluation4 = evaluation_generator.flow_from_directory(
            validation_files,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False,
            subset='validation')

    if classification ==False:
        Y_pred = model.predict(evaluation4)
        y_pred=[]
        for i in range(0,len(Y_pred)):
            n=float(Y_pred[i])
            y_pred.append(round(n))
        print()
        print('Confusion Matrix')
        print(confusion_matrix(evaluation4.classes, y_pred))
        print('Classification Report')
        target_names = ['0', '1', '2','3','4']
        print(classification_report(evaluation4.classes, y_pred, target_names=target_names))
        print('MAE:')
        acc=0
        for i in range(0,len(evaluation4.classes)):
            YR=evaluation4.classes[i]
            YP=y_pred[i]
            acc=acc+abs(YR-YP)
        print(acc/len(evaluation4.classes))
        print('KAPPA:')
        kappa = cohen_kappa_score(y_pred,evaluation4.classes)
        print(kappa)
        print('MSE:')
        mse=mean_squared_error(evaluation4.classes,y_pred)
        print(mse)
    else:
        Y_pred = model.predict(evaluation4)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(evaluation4.classes, y_pred))
        print('Classification Report')
        target_names = ['0', '1', '2', '3', '4']
        print(classification_report(evaluation4.classes, y_pred, target_names=target_names))

        print('KAPPA:')
        kappa = cohen_kappa_score(y_pred, evaluation4.classes)
        print(kappa)

testModel()