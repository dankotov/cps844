# Ivan Lytovka, 500861433
# Danyil Kotov, 500877422

import time
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

import matplotlib.style as style

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

# print(data)

targetClass = data["Class"]
attributeClass = data.iloc[:, 1:]

dataTrain, dataTest, classTrain, classTest = train_test_split(
    attributeClass, targetClass, test_size=0.3, stratify=targetClass, random_state=1)

classifiers = {
    # "DT (default - max_depth=None)": {
    #     "clf":tree.DecisionTreeClassifier(),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "DT (max_depth=3)": {
    #     "clf":tree.DecisionTreeClassifier(criterion='entropy', max_depth=3),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "DT (max_depth=2)": {
    #     "clf":tree.DecisionTreeClassifier(criterion='entropy', max_depth=2),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "KN (default - n_neighbors=5)": {
    #     "clf":KNeighborsClassifier(),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "KN (n_neighbors=10)": {
    #     "clf":KNeighborsClassifier(n_neighbors=10),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "KN (n_neighbors=100)": {
    #     "clf":KNeighborsClassifier(n_neighbors=100),
    #     "accuracy":0,
    #     "runtime":0,
    # },   
    # "SV (default - kernel=radial basis)": {
    #     "clf":SupportVectorClassifier(),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "SV (kernel=linear)": {
    #     "clf":SupportVectorClassifier(kernel='linear'),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "SV (kernel=polynomial)": {
    #     "clf":SupportVectorClassifier(kernel='poly'),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "SV (kernel=sigmoid)": {
    #     "clf":SupportVectorClassifier(kernel='sigmoid'),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "RF (default - n_estimators=100)": {
    #     "clf":RandomForestClassifier(),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "RF (default - n_estimators=800)": {
    #     "clf":RandomForestClassifier(n_estimators=800),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "RF (default - n_estimators=50)": {
    #     "clf":RandomForestClassifier(n_estimators=50),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    # "RF (default - n_estimators=10)": {
    #     "clf":RandomForestClassifier(n_estimators=5),
    #     "accuracy":0,
    #     "runtime":0,
    # },
    "MLP (default - hidden_layer_sizes=(100,)": {
        "clf":MLPClassifier(random_state=5),
        "accuracy":0,
        "runtime":0,
    },
     "MLP (default - hidden_layer_sizes=(3,3,3)": {
        "clf":MLPClassifier(hidden_layer_sizes=(3,3,3),random_state=5),
        "accuracy":0,
        "runtime":0,
    },
    "MLP (default - hidden_layer_sizes=(8,8,8)": {
        "clf":MLPClassifier(hidden_layer_sizes=(8,8,8),random_state=5),
        "accuracy":0,
        "runtime":0,
    },
}

for clf_tag in classifiers:
    clf = classifiers[clf_tag]["clf"]

    start = time.time()

    clf.fit(dataTrain, classTrain)
    clf_prediction = clf.predict(dataTest)

    end = time.time()

    accuracy = accuracy_score(classTest, clf_prediction)
    run_time = end - start


    classifiers[clf_tag]["accuracy"] = accuracy
    classifiers[clf_tag]["runtime"] = run_time

    print(f'{clf_tag} | Accuracy: {accuracy}; Time: {run_time}s')


