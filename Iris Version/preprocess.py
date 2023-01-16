import numpy as np
import pandas as pd

def preprocess(): # Function to Read dataset and count Male and Female in each class to replace Null values with the
    # one which has more occured in that class
    # after that replace Male with 1 and Female with -1
    df = pd.read_csv("Iris.csv")
    setosaClass = df[(df['Species'] == 'Iris-setosa')]
    versicolorClass = df[(df['Species'] == 'Iris-versicolor')]
    virginicaClass = df[(df['Species'] == 'Iris-virginica')]
    setosaClass.replace("Iris-setosa", "100", inplace=True) # [1, 0, 0]
    versicolorClass.replace("Iris-versicolor", "010", inplace=True)
    virginicaClass.replace("Iris-virginica", "001", inplace=True)
    return setosaClass, versicolorClass, virginicaClass


def fitdata():
    SClass, VCClass, VClass = preprocess()
    # Train Data
    TrainSet = SClass[:30]
    TrainSet = TrainSet.append(VCClass[:30], ignore_index=True)
    TrainSet = TrainSet.append(VClass[:30], ignore_index=True)
    TrainLables = TrainSet["Species"]
    TrainData = TrainSet.drop(columns=["Species","Id"])
    # Test Data
    TestSet = SClass[30:]
    TestSet = TestSet.append(VCClass[30:], ignore_index=True)
    TestSet = TestSet.append(VClass[30:], ignore_index=True)
    TestLabels = TestSet["Species"]
    TestData = TestSet.drop(columns=["Species","Id"])
    return TrainLables, TrainData, TestLabels, TestData