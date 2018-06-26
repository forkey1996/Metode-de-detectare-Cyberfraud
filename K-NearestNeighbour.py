import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import numpy as npy
import time
import dataProcessing

dataTrain, dataTest, targetTrain, targetTest = dataProcessing.readDataAndUndersample()
dataTrain = dataTrain.values
dataTest = dataTest.values
targetTrain = targetTrain.values
targetTest = targetTest.values

features_number = 30
k = 3

data = tf.placeholder(dtype=tf.float32, shape=[None, features_number], name="data")
dataForTest = tf.placeholder(dtype=tf.float32, shape=[None, features_number], name="dataForTest")
target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target")

distance = tf.reduce_sum(tf.abs(tf.add(data, tf.negative(tf.expand_dims(dataForTest,1)))), axis=2)
_, topKIndices = tf.nn.top_k(tf.negative(distance), k=k)
topKLabel = tf.gather(target, topKIndices)
predictionSum = tf.reduce_sum(topKLabel, axis=1)
prediction = tf.to_float(tf.greater_equal(predictionSum, 3))

originalData, originalTarget = dataProcessing.readOriginalData()
originalData = originalData.values
originalTarget = originalTarget.values

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = time.time()
    predictedValues = sess.run(prediction, feed_dict={data: dataTrain, target: npy.matrix(targetTrain).T, dataForTest: originalData[1:50000]})
    predictedValues2 = sess.run(prediction,
                                feed_dict={data: dataTrain, target: npy.matrix(targetTrain).T, dataForTest: originalData[50000:100000]})
    predictedValues3 = sess.run(prediction,
                                feed_dict={data: dataTrain, target: npy.matrix(targetTrain).T, dataForTest: originalData[100000:150000]})
    predictedValues4 = sess.run(prediction,
                                feed_dict={data: dataTrain, target: npy.matrix(targetTrain).T, dataForTest: originalData[150000:200000]})
    predictedValues5 = sess.run(prediction,
                                feed_dict={data: dataTrain, target: npy.matrix(targetTrain).T, dataForTest: originalData[200000:250000]})
    predictedValues6 = sess.run(prediction,
                                feed_dict={data: dataTrain, target: npy.matrix(targetTrain).T, dataForTest: originalData[250000:len(originalData)]})
    predictedValues = npy.concatenate((predictedValues, predictedValues2, predictedValues3, predictedValues4, predictedValues5, predictedValues6))

    end = time.time()
    print("Timp de rulare pentru intreg setul de date, folosind ca intrare 80% din setul de date subesantionat: ", end - start)

    confusionMatrix = tf.confusion_matrix(originalTarget[1:len(originalData)], predictedValues, 2)
    confusionMatrixResult = sess.run(confusionMatrix)
    sbn.heatmap(confusionMatrixResult, annot=True, fmt='g')
    plt.ylabel('Valori corecte')
    plt.xlabel('Valori prezise')
    plt.show()

    falsePositiveRate, truePositiveRate, _ = roc_curve(originalTarget[1:len(originalData)], predictedValues)

    dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
    aucScore = roc_auc_score(originalTarget[1:len(originalData)], predictedValues)
    print('Scor AUC: ', aucScore)