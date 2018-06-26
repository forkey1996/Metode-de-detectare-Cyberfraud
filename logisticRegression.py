import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import time
from sklearn.linear_model import LogisticRegression

import dataProcessing

dataTrain, dataTest, targetTrain, targetTest = dataProcessing.readDataAndUndersample()
start = time.time()
logisticRegression = LogisticRegression(C=0.01, penalty='l1')
logisticRegression.fit(dataTrain, targetTrain)
end = time.time()
print("Timp de antrenare pe 80% din setul de date subesantionat: ", end - start)

start = time.time()
targetUnderSamplePrediction = logisticRegression.predict_proba(dataTest)[:, 1] > 0.75
end = time.time()
print("Timp de rulare pentru 20% din setul de date subesantionat date: ", end - start)

confusionMatrix = confusion_matrix(targetTest, targetUnderSamplePrediction)
sbn.heatmap(confusionMatrix, annot=True, fmt='g')
plt.ylabel('Valori corecte')
plt.xlabel('Valori prezise')
plt.show()

falsePositiveRate, truePositiveRate, _ = roc_curve(targetTest, targetUnderSamplePrediction)

dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
aucScore = roc_auc_score(targetTest, targetUnderSamplePrediction)
print('Scor AUC: ', aucScore)

dataOriginal, targetOriginal = dataProcessing.readOriginalData()

start = time.time()
targetOriginalPrediction = logisticRegression.predict_proba(dataOriginal)[:, 1] > 0.75
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