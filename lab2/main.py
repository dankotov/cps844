import pandas as pd
import numpy as np

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

# count the number of NaN occurences in each column
col_nan_count = data.apply(pd.value_counts, dropna=False).loc[np.nan]

# dropping all rows containing missing values
data.dropna(inplace=True)

# converting all 
data["bare nuclei"] = pd.to_numeric(data["bare nuclei"])

# "marginal adhesion", "single epithelial cell size", "bland chromatin", "normal nucleoli", "mitoses" have outliers
data.boxplot()

# number of duplicate entries
duplicated_count = data.duplicated().value_counts()[True]

data.drop_duplicates(inplace=True)

data.hist(column="clump thickness")

data["clump thickness"] = pd.cut(data["clump thickness"], bins=4)
# (0.991, 3.25] 131
# (3.25, 5,5]   140
# (5.5, 7.75]   52
# (7.75, 10.0]  126
category_ranges = data["clump thickness"].value_counts(sort=False) 

sample = data.sample(frac=0.01, replace=False, random_state=1)






