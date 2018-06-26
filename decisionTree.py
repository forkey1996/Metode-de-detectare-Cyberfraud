import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import time
from sklearn.tree import DecisionTreeClassifier

import dataProcessing

dataTrain, dataTest, targetTrain, targetTest = dataProcessing.readDataAndUndersample()
start = time.time()
decisionTree = DecisionTreeClassifier(criterion="entropy")
decisionTree.fit(dataTrain, targetTrain)
end = time.time()
print("Timp de antrenare pe 80% din setul de date subesantionat: ", end - start)

start = time.time()
targetUnderSamplePrediction = decisionTree.predict_proba(dataTest)[:, 1] > 0.75
end = time.time()
print("Timp de antrenare pe 20% din setul de date subesantionat: ", end - start)

confusionMatrix = confusion_matrix(targetTest,targetUnderSamplePrediction)
sbn.heatmap(confusionMatrix, annot=True, fmt='g')
plt.ylabel('Valori corecte')
plt.xlabel('Valori prezise')
plt.show()

falsePositiveRate, truePositiveRate, _ = roc_curve(targetTest, targetUnderSamplePrediction)

dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
aucScore = roc_auc_score(targetTest, targetUnderSamplePrediction)
print('Scor AUC: ', aucScore)

dataOriginalTrain, dataOriginalTest, targetOriginalTrain, targetOriginalTest = dataProcessing.readOriginalDataAndSplit()
start = time.time()
decisionTree.fit(dataOriginalTrain, targetOriginalTrain)
end = time.time()
print("Timp de antrenare pe 80% din date: ", end - start)

start = time.time()
targetOriginalPrediction = decisionTree.predict_proba(dataOriginalTest)[:,1] > 0.25
end = time.time()
print("Timp de rulare pentru 20% din date: ", end - start)
confusionMatrix = confusion_matrix(targetOriginalTest ,targetOriginalPrediction)
sbn.heatmap(confusionMatrix, annot=True, fmt='g')
plt.ylabel('Valori corecte')
plt.xlabel('Valori prezise')
plt.show()

falsePositiveRate, truePositiveRate, _ = roc_curve(targetOriginalTest, targetOriginalPrediction)

dataProcessing.rocPlot(falsePositiveRate, truePositiveRate)
aucScore = roc_auc_score(targetOriginalTest, targetOriginalPrediction)
print('Scor AUC: ', aucScore)