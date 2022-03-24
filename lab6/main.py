import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from apyori import apriori

# ------- PART 1 -------
data = pd.read_csv("./weather.csv")

# converting categorical to dummy
data = pd.get_dummies(data=data, dtype='float64')

# dropping play_no column
data.drop(columns='play_no', inplace=True)

# separating attributes from targe 
targetClassData = data['play_yes']
attributeData = data.iloc[:,:-1]

# initializing and training the classifier
clf = GaussianNB()
clf.fit(attributeData, targetClassData)

# predicting probability 
probability = clf.predict_proba([[66,90,True,0,0,1]])
# printing out probability of play=no and play=yes
print(f'Probability of play=no : {probability[0][0]*100}%; play=yes : {probability[0][1]*100}%\n\n')

# ------- PART 2 -------
data = pd.read_csv("./weather.csv")

# discretizing temperature data
data['temperature'] = pd.cut(x=data['temperature'], bins=3, labels=['cool', 'mild', 'hot'])
# discretizing humidity data
data['humidity'] = pd.cut(x=data['humidity'], bins=2, labels=['normal', 'high'])

# converting booleans to strings for column 'windy'
data['windy'] = data['windy'].apply(lambda x: 'true' if x == True else 'false')

# converting dataset to list of lists
weather_records = []
for i in range(data.shape[0]):
    weather_records.append(data.iloc[i].tolist())

# running apriori 
rules = apriori(weather_records, min_support=0.28, min_confidence=0.5)
    
# printing out generated rules with respective support and confidence levels
print("Rules:")
for rule in rules:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        'Support:',rule.support, 'Confidence:', rule.ordered_statistics[0].confidence, 'Lift:', rule.ordered_statistics[0].lift)
    
# After playing around with values for min_suppport and min_confidence, we can conclude that the lower we make the thresholds,
# the more rules we will be able to get. I also decided to print out the lift to understand, which rules in fact provide a sound
# proof of association. 