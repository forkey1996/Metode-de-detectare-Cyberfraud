import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
import numpy as npy
import time

import dataProcessing

dataTrain, dataTest, targetTrain, targetTest = dataProcessing.readDataAndUndersample()

learningRate = 0.01
featuresNumber = 30
trainEpochs = 10000

data = tf.placeholder(dtype=tf.float32, shape=[None, featuresNumber], name="data")
target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target")

weights = tf.get_variable("weights", shape=[featuresNumber, 1], initializer=tf.contrib.layers.xavier_initializer())
bias = tf.get_variable("bias", [1], initializer=tf.zeros_initializer())
model = tf.add(tf.matmul(data, weights), bias)

prediction = tf.nn.sigmoid(model)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=target))
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = time.time()
    for epoch in range(trainEpochs):
        batchSize = 128
        totalBatch = int(len(dataTrain) / batchSize)
        sess.run([optimizer, cost], feed_dict={data: dataTrain, target: npy.matrix(targetTrain).T})
    end = time.time()
    print("Timp de antrenare pe 80% din setul de date subesantionat: ", end - start)

    correctPrediction = tf.equal(tf.to_float(tf.greater(prediction, 0.75)), target)
    targetUnderSamplePrediction = npy.empty(shape=[0,1])
    start = time.time()
    targetUnderSamplePrediction = sess.run(tf.to_float(tf.greater(prediction, 0.4)),feed_dict={data: dataTest, target: npy.matrix(targetTest).T})
    end = time.time()
    print("Timp de antrenare pe 20% din setul de date subesantionat: ", end - start)

    confusionMatrix = tf.confusion_matrix(targetTest, targetUnderSamplePrediction,2)
    confusionMatrixResult = sess.run(confusionMatrix)
    sbn.heatmap(confusionMatrixResult, annot=True, fmt='g')
    plt.ylabel('Valori corecte')
    plt.xlabel('Valori prezise')
    plt.show()

    falsePositiveRate, truePositiveRate, _ = roc_curve(targetTest, targetUnderSamplePrediction)

    dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
    aucScore = roc_auc_score(targetTest, targetUnderSamplePrediction)
    print('Scor AUC: ', aucScore)


    dataOriginal, targetOriginal = dataProcessing.readOriginalData()
    start = time.time()
    targetOriginalPrediction = sess.run(tf.to_float(tf.greater(prediction, 0.75)), feed_dict={data: dataOriginal, target: npy.matrix(targetOriginal).T})
    end = time.time()
    print("Timp de rulare pentru intreg setul de date: ", end - start)
    confusionMatrix = confusion_matrix(targetOriginal, targetOriginalPrediction)
    sbn.heatmap(confusionMatrix, annot=True, fmt='g')
    plt.ylabel('Valori corecte')
    plt.xlabel('Valori prezise')
    plt.show()

    falsePositiveRate, truePositiveRate, _ = roc_curve(targetOriginal, targetOriginalPrediction)

    dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
    aucScore = roc_auc_score(targetOriginal, targetOriginalPrediction)
    print('Scor AUC: ', aucScore)