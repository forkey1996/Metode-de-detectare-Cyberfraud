import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import time
import dataProcessing

dataTrain, dataTest, targetTrain, targetTest = dataProcessing.readDataAndUndersample()
dataTrain = dataTrain.values
dataTest = dataTest.values
targetTrain = targetTrain.values
targetTest = targetTest.values

features_number = 30

data = tf.placeholder(dtype=tf.float32, shape=[None, features_number], name="data")
dataForTest = tf.placeholder(dtype=tf.float32, shape=[features_number], name="dataForTest")
target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target")

distance = tf.reduce_sum(tf.abs(tf.add(data, tf.negative(dataForTest))), reduction_indices=1)
prediction = tf.arg_min(distance, 0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    predictionsList = []
    start = time.time()
    for idx in range(len(dataTest)):
        predictedIndex = sess.run(prediction, feed_dict={data: dataTrain, dataForTest: dataTest[idx, :]})
        predictionsList.append(targetTrain[predictedIndex])
    end = time.time()
    print("Timp de rulare pentru intreg setul de date: ", end - start)

    confusionMatrix = tf.confusion_matrix(targetTest, predictionsList,2)
    confusionMatrixResult = sess.run(confusionMatrix)
    sbn.heatmap(confusionMatrixResult, annot=True, fmt='g')
    plt.ylabel('Valori corecte')
    plt.xlabel('Valori prezise')
    plt.show()

    falsePositiveRate, truePositiveRate, _ = roc_curve(targetTest, predictionsList)

    dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
    aucScore = roc_auc_score(targetTest, predictionsList)
    print('Scor AUC: ', aucScore)