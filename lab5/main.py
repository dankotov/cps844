from statistics import mean
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


headers = [
    "sample code #",
    "clump thickness",
    "cell size uniformity",
    "cell shape uniformity",
    "marginal adhesion",
    "single epithelial cell size",
    "bare nuclei",
    "bland chromatin",
    "normal nucleoli",
    "mitoses",
    "class",
]

# reading CSV, assigning headers
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=headers)

# dropping sample code # column
data.drop(labels="sample code #", axis=1, inplace=True)

# replace ? values with numpy NaN
data.replace(to_replace="?", value=np.nan, inplace=True)

# dropping all rows containing missing values
data.dropna(inplace=True)

# converting all 
data["bare nuclei"] = pd.to_numeric(data["bare nuclei"])

# drop duplicates 
data.drop_duplicates(inplace=True)

# separating features from target class
classData = data["class"]
attributeData = data.iloc[:,0:-1]

# mapping class label values
classData = classData.map(lambda x: 0 if x == 2 else 1)

# constructing a NearestNeighbors classifier
clf = KNeighborsClassifier()

# assesing various scores using 10-fold cross-validation
accuracy = cross_val_score(clf, attributeData, classData, cv=10, scoring = 'accuracy')
f1 = cross_val_score(clf, attributeData, classData, cv=10, scoring = 'f1')
precision = cross_val_score(clf, attributeData, classData, cv=10, scoring = 'precision')
recall = cross_val_score(clf, attributeData, classData, cv=10, scoring = 'recall')

# printing average values of computed scores
print(f'Average\n\tAccuracy: {mean(accuracy)}\n\tf1: {mean(f1)}\n\tPrecision: {mean(precision)}\n\tRecall: {mean(recall)}')

# splitting the data
dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, test_size=0.3, stratify=classData, random_state=1)

# training the classifier
clf.fit(dataTrain, classTrain)

# making a prediction
prediction = clf.predict(dataTest)

# computing a confusion matrix
c_matrix = confusion_matrix(classTest, prediction)

# printing confusion matrix
print(f'\nConfusion matrix:\n{c_matrix}')









