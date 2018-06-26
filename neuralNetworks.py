import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import numpy as npy
import time

import dataProcessing

dataTrain, dataTest, targetTrain, targetTest = dataProcessing.readDataAndUndersampleSeparateClass()
learningRate = 0.01
featuresNumber = 30
trainEpochs = 10000
classNumber = 2
multiply = 2
hiddenNodes1 = 32
hiddenNodes2 = round(hiddenNodes1 * multiply)
hiddenNodes3 = round(hiddenNodes2 * multiply)

data = tf.placeholder(dtype=tf.float32, shape=[None, featuresNumber])

weights1 = tf.Variable(tf.truncated_normal([featuresNumber, hiddenNodes1]))
bias1 = tf.Variable(tf.zeros([hiddenNodes1]))
layer1 = tf.add(tf.matmul(data, weights1), bias1)

weights2 = tf.Variable(tf.truncated_normal([hiddenNodes1, hiddenNodes2]))
bias2 = tf.Variable(tf.zeros([hiddenNodes2]))
layer2 = tf.add(tf.matmul(layer1, weights2), bias2)

weights3 = tf.Variable(tf.truncated_normal([hiddenNodes2, hiddenNodes3]))
bias3 = tf.Variable(tf.zeros([hiddenNodes3]))
layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights3), bias3))

weights4 = tf.Variable(tf.truncated_normal([hiddenNodes3, classNumber]))
bias4 = tf.Variable(tf.zeros([classNumber]))
layerOut = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights4), bias4))

model = layerOut
target = tf.placeholder(tf.float32, [None, classNumber])

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=target))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

correctPrediction = tf.equal(tf.argmax(model,1), tf.argmax(target,1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    start = time.time()
    for epoch in range(trainEpochs):
        sess.run([optimizer], feed_dict={data: dataTrain, target: targetTrain})
    end = time.time()
    print("Timp de antrenare: ", end-start)

    prediction = model[:,1]>0.75
    testingPredictions = sess.run([prediction], feed_dict={data: dataTest, target: targetTest})
    testingPredictions = npy.asarray(testingPredictions)
    testingPredictions = testingPredictions[0]
    confusionMatrix = tf.confusion_matrix(targetTest.Fraud, testingPredictions,2)
    confusionMatrixResult = sess.run(confusionMatrix)
    sbn.heatmap(confusionMatrixResult, annot=True, fmt='g')
    plt.ylabel('Valori corecte')
    plt.xlabel('Valori prezise')
    plt.show()
    falsePositiveRate, truePositiveRate, _ = roc_curve(targetTest.Fraud, testingPredictions)
    dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
    aucScore = roc_auc_score(targetTest.Fraud, testingPredictions)
    print('Scor AUC: ', aucScore)

    originalData, originalTarget = dataProcessing.readOriginalDataSeparatedClass()
    start = time.time()
    testingPredictions = sess.run([prediction], feed_dict={data: originalData, target: originalTarget})
    end = time.time()
    print("Timp de rulare pe tot setul de date: ", end-start)
    testingPredictions = npy.asarray(testingPredictions)
    testingPredictions = testingPredictions[0]
    confusionMatrix = tf.confusion_matrix(originalTarget.Fraud, testingPredictions,2)
    confusionMatrixResult = sess.run(confusionMatrix)
    sbn.heatmap(confusionMatrixResult, annot=True, fmt='g')
    plt.ylabel('Valori corecte')
    plt.xlabel('Valori prezise')
    plt.show()
    falsePositiveRate, truePositiveRate, _ = roc_curve(originalTarget.Fraud, testingPredictions)
    dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
    aucScore = roc_auc_score(originalTarget.Fraud, testingPredictions)
    print('Scor AUC: ', aucScore)