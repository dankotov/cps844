# Ivan Lytovka, 500861433
# Danyil Kotov, 500877422

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# classifiers
from sklearn import tree
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("./Dry_Bean_Dataset.csv")

headers = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
    "Class"
]

# categorical to numerical
labelEncoder = LabelEncoder()

data["Class"] = labelEncoder.fit_transform(data["Class"])

print(data)

targetClass = data["Class"]
attributeClass = data.iloc[:, 1:]

dataTrain, dataTest, classTrain, classTest = train_test_split(
    attributeClass, targetClass, test_size=0.3, stratify=targetClass, random_state=1)

classifiers = {
    "Decision Tree":tree.DecisionTreeClassifier(), 
    "K Neighbors":KNeighborsClassifier(),
    "Support Vector":SupportVectorClassifier(),
    "Random Forest":RandomForestClassifier(),
    "ML":MLPClassifier()
}

for clf in classifiers:
    classifiers[clf].fit(dataTrain, classTrain)
    clf_prediction = classifiers[clf].predict(dataTest)
    print(f'The accuracy of the {clf} classifier is {accuracy_score(classTest, clf_prediction)}')
