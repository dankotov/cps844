import pandas as pd

headers = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "class",    
]

data = pd.read_csv("iris.data", names=headers)

# average
print("Average")
print(data.mean(numeric_only=True))
print("\n")

# standard deviation
print("Standard Deviation")
print(data.std(numeric_only=True))
print("\n")

# min
print("Min value")
print(data.min(numeric_only=True))
print("\n")

# max 
print("Max value")
print(data.max(numeric_only=True))
print("\n")

# class frequency
print("Class Frequency")
print(data["class"].value_counts())