# -*- coding: utf-8 -*-
"""
Part 1
"""

import pandas as pd
from scipy.cluster import hierarchy

# (0 point) Import the vertebrate.csv data
data = pd.read_csv('./vertebrate.csv')

# (5 points) Pre-process data: create a new variable and bind it with all the numerical attributes (i.e. all except the 'Name' and 'Class')
NumericalAttributes = data.drop(data.select_dtypes(exclude="number"), axis=1)

### (10 points) Single link (MIN) analysis + plot associated dendrogram ###
min_analysis = hierarchy.linkage(NumericalAttributes, 'single')

# # (5 points) Plot the associated dendrogram. 
# # Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# # Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
dn_single = hierarchy.dendrogram(min_analysis, labels=data['Name'].tolist(), orientation='right')

# ### (10 points) Complete Link (MAX) analysis + plot associated dendrogram ###
max_analysis = hierarchy.linkage(NumericalAttributes, 'complete')

# # (5 points) Plot the associated dendrogram. 
# # Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# # Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
dn_complete = hierarchy.dendrogram(max_analysis, labels=data['Name'].tolist(), orientation='right')

# ### (10 points) Group Average analysis ###
average_analysis = hierarchy.linkage(NumericalAttributes, 'average')

# # (5 points) Plot the associated dendrogram. 
# # Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# # Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
dn_average = hierarchy.dendrogram(average_analysis, labels=data['Name'].tolist(), orientation='right')