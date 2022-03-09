import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# 1 and 3
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']
data.drop(['Sample code number'],axis=1, inplace=True)
data.replace('?', np.NaN, inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# 4
data['Class'].replace(4, 1, inplace=True)
data['Class'].replace(2, 0, inplace=True)

# 5
targetClass = data["Class"]
attributeClass = data.iloc[:, 1:-1]

clf = KNeighborsClassifier(n_neighbors=5)
Xtrain, Xtest, ytrain, ytest  = train_test_split(attributeClass, targetClass, test_size=0.4)

clf = clf.fit(Xtrain, ytrain)

#6

# clf = clf.fit(attributeClass, targetClass)
# print(clf)

score_accuracy = cross_val_score(clf, attributeClass, targetClass, cv=5, scoring = 'accuracy')
score_f1 = cross_val_score(clf, attributeClass, targetClass, cv=5, scoring = 'f1')
score_precision = cross_val_score(clf, attributeClass, targetClass, cv=5, scoring = 'precision')
score_macro = cross_val_score(clf, attributeClass, targetClass, cv=5, scoring = 'recall_macro')

# print(score)
