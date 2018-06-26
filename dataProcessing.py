import numpy as npy
import pandas as pnd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
dataLocation = 'input/creditcard.csv'

# Citeste date subesantionate
def readDataAndUndersample():
    data = pnd.read_csv(dataLocation)

    # Redimensionare date
    robustScaler = RobustScaler()

    scaledAmount = robustScaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    scaledTime = robustScaler.fit_transform(data['Time'].values.reshape(-1, 1))

    data.drop(['Time', 'Amount'], axis=1, inplace=True)
    data.insert(0, 'scaledAmount', scaledAmount)
    data.insert(1, 'scaledTime', scaledTime)

    # Creare set de date subesantionat
    data = data.sample(frac=1)
    frauds = data[data['Class'] == 1]
    normals = data[data['Class'] == 0][:len(frauds)]

    equalDistributedData = pnd.concat([frauds, normals])

    randomEqualDistributedData = equalDistributedData.sample(frac=1, random_state=42)

    # Impartim datele in date de antrenare si de test
    data = randomEqualDistributedData.drop('Class', axis=1)
    target = randomEqualDistributedData['Class']

    return train_test_split(data, target, test_size=0.2, random_state=42)


def readDataAndUndersampleSeparateClass():
    data = pnd.read_csv(dataLocation)

    # Redimensionare date
    robustScaler = RobustScaler()

    scaledAmount = robustScaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    scaledTime = robustScaler.fit_transform(data['Time'].values.reshape(-1, 1))

    data.drop(['Time', 'Amount'], axis=1, inplace=True)
    data.insert(0, 'scaledAmount', scaledAmount)
    data.insert(1, 'scaledTime', scaledTime)

    data.loc[data.Class == 0, 'Normal'] = 1
    data.loc[data.Class == 1, 'Normal'] = 0

    data = data.rename(columns={'Class': 'Fraud'})

    # Creare set de date subesantionat
    data = data.sample(frac=1)
    frauds = data[data['Fraud'] == 1]
    normals = data[data['Fraud'] == 0][:len(frauds)]

    equalDistributedData = pnd.concat([frauds, normals])

    randomEqualDistributedData = equalDistributedData.sample(frac=1, random_state=42)

    # Impartim datele in date de antrenare si de test
    dataResult = randomEqualDistributedData.drop(['Normal', 'Fraud'], axis=1)
    target = randomEqualDistributedData['Normal']
    target = pnd.concat([target, randomEqualDistributedData['Fraud']], axis=1)
    return train_test_split(dataResult, target, test_size=0.2, random_state=42)


def readOriginalData():
    data = pnd.read_csv(dataLocation)

    # Redimensionare date
    robustScaler = RobustScaler()

    scaledAmount = robustScaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    scaledTime = robustScaler.fit_transform(data['Time'].values.reshape(-1, 1))

    data.drop(['Time', 'Amount'], axis=1, inplace=True)
    data.insert(0, 'scaledAmount', scaledAmount)
    data.insert(1, 'scaledTime', scaledTime)

    parameters = data.drop('Class', axis=1)
    target = data ['Class']
    return parameters, target


def readOriginalDataSeparatedClass():
    data = pnd.read_csv(dataLocation)
    # Redimensionare date
    robustScaler = RobustScaler()

    scaledAmount = robustScaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    scaledTime = robustScaler.fit_transform(data['Time'].values.reshape(-1, 1))

    data.drop(['Time', 'Amount'], axis=1, inplace=True)
    data.insert(0, 'scaledAmount', scaledAmount)
    data.insert(1, 'scaledTime', scaledTime)

    data.loc[data.Class == 0, 'Normal'] = 1
    data.loc[data.Class == 1, 'Normal'] = 0

    data = data.rename(columns={'Class': 'Fraud'})

    dataResult = data.drop(['Normal', 'Fraud'], axis=1)
    target = data['Normal']
    target = pnd.concat([target, data['Fraud']], axis=1)
    return dataResult, target


def readOriginalDataAndSplit():
    data = pnd.read_csv(dataLocation)
    # Redimensionare date
    robustScaler = RobustScaler()

    scaledAmount = robustScaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    scaledTime = robustScaler.fit_transform(data['Time'].values.reshape(-1, 1))

    data.drop(['Time', 'Amount'], axis=1, inplace=True)
    data.insert(0, 'scaledAmount', scaledAmount)
    data.insert(1, 'scaledTime', scaledTime)

    #Impartire in date de test si de antrenare
    parameters = data.drop('Class', axis=1)
    target = data ['Class']
    return train_test_split(parameters, target, test_size=0.2, random_state=42)


def rocPlot(falsePositiveRate, truePositiveRate):
    plt.title('Curba ROC')
    plt.plot(falsePositiveRate, truePositiveRate, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Rata rezultatelor pozitive greșite')
    plt.ylabel('Rata rezultatelor pozitive corecte')
    plt.axis([-0.01, 1, 0, 1])
    plt.show()


def showDataInfo():
    data = pnd.read_csv(dataLocation)
    print(data.head())
    print(data.describe())
    print('Valori nule:', data.isnull().sum().max())
    print(data.columns)

    paletteColors = ["#0102DD", "#DF0200"]
    sbn.countplot('Class', data=data, palette=paletteColors)
    plt.title('Distribuția claselor \n (0: Normal || 1: Fraudă)', fontsize=14)
    plt.show()

    plt.subplots(figsize=(12, 6))
    sbn.regplot(x='Time', y='Amount', data=data[data['Class'] == 1], fit_reg=False, scatter_kws={'s': 5})
    plt.title('Tranzacțiile Fraudulente Valoare vs Timp')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.show()

    plt.subplots(figsize=(14, 6))
    sbn.regplot(x='Time', y='Amount', data=data[data['Class'] == 0], fit_reg=False, scatter_kws={'s': 5})
    plt.title('Tranzacțiile Normale Valoare vs Timp')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.show()

    columns = data.columns.values

    i = 0
    normals = data[data['Class'] == 0]
    frauds = data[data['Class'] == 1]

    sbn.set_style('whitegrid')
    plt.subplots(3, 4, figsize=(16, 28))

    for feature in columns:
        i += 1
        if i <= 12:
            plt.subplot(3, 4, i)
            sbn.kdeplot(normals[feature], label="Class = 0")
            sbn.kdeplot(frauds[feature], label="Class = 1")
            plt.xlabel(feature, fontsize=12)
            plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

    i = 0

    sbn.set_style('whitegrid')
    plt.subplots(3, 4, figsize=(16, 28))

    for feature in columns:
        i += 1
        if i > 12 and i <=24:
            plt.subplot(3, 4, i - 12)
            sbn.kdeplot(normals[feature], label="Class = 0")
            sbn.kdeplot(frauds[feature], label="Class = 1")
            plt.xlabel(feature, fontsize=12)
            plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

    i = 0

    sbn.set_style('whitegrid')
    plt.subplots(2, 3, figsize=(16, 28))

    for feature in columns:
        i += 1
        if i > 24 and i < 31:
            plt.subplot(3, 3, i - 24)
            sbn.kdeplot(normals[feature], label="Class = 0")
            sbn.kdeplot(frauds[feature], label="Class = 1")
            plt.xlabel(feature, fontsize=12)
            plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

showDataInfo()