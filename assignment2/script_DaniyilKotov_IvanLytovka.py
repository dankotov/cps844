import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from apyori import apriori


dataClustering = pd.read_csv("./Dry_Bean_Dataset.csv")

# dropping na values 
dataClustering.dropna(inplace=True)
# storing dataset classes separately for later reference
dataClusteringClass = dataClustering["Class"]
# storing the list of possible classes in the dataset
datasetClusteringClasses = dataClustering["Class"].unique()
# separating attributes from classes
dataClusteringAttributes = dataClustering.iloc[:, :-1]


sses = {}

# testing a range of k cluster numbers while computing their SSE scores to later use in the elbow method
for k in range(1, dataClusteringClass.nunique()+1):
    kmeans = KMeans(n_clusters=k, n_init=100).fit(dataClusteringAttributes)
    sses[k] = kmeans.inertia_

# plotting SSE to cluster number graph for the elbow method
plt.figure()
plt.plot(list(sses.keys()), list(sses.values()))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Errors")
plt.show()

# according to the plotted sse graph, we concluded the elbow to be at k=4
kmeans = KMeans(n_clusters=4, n_init=100).fit(dataClusteringAttributes)
# extracting cluster labels provided by the KMeans algorithm
clustersLabels = pd.DataFrame(kmeans.labels_,columns=['Cluster ID'])

# concatenating the cluster labels to the original dataset
dataClustering = pd.concat([dataClustering, clustersLabels], axis=1)

# looping through each cluster label
for label in clustersLabels["Cluster ID"].unique():
    # extract all the rows that were labeled this label by the KMeans algorithm
    cluster = dataClustering.loc[dataClustering["Cluster ID"] == label]
    # count the number of different elements of various classes present in the resulting cluster
    classesInCluster = cluster["Class"].value_counts()
    # extracting the most common class in cluster
    mostCommonClassInCluster = cluster["Class"].mode()[0]
    # calculating purity of the cluster
    purity = classesInCluster[mostCommonClassInCluster] / len(cluster.index)
    # building value array for the bar graphs of the number of rows of each class in the cluster
    classCounts = []
    for datasetClusteringClass in datasetClusteringClasses:
        if datasetClusteringClass in classesInCluster:
            classCounts.append(classesInCluster[datasetClusteringClass])
        else:
            classCounts.append(0)
    plt.bar(x=datasetClusteringClasses, height=classCounts)
    plt.title(label=f"Cluster {label}")
    plt.show()
    print(f"Cluster ID: {label}, Most Common Class In Cluster: {mostCommonClassInCluster}, Purity: {purity}")
    

headers = [
    "class",
    "handicapped-infants",
    "water-project-cost-sharing",
    "adoption-of-the-budget-resolution",
    "physician-fee-freeze",
    "el-salvador-aid",
    "religious-groups-in-schools",
    "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras",
    "mx-missile",
    "immigration",
    "synfuels-corporation-cutback",
    "education-spending",
    "superfund-right-to-sue",
    "crime",
    "duty-free-exports",
    "export-administration-act-south-africa"]

dataAssociation = pd.read_csv('./house-votes-84.data', names=headers)

# altering the values of the attributes from the dataset as described in the report
for columnName in dataAssociation:
    dataAssociation[columnName] = dataAssociation[columnName].apply(lambda attribute: f"{columnName}_{attribute}" if attribute != "?" else np.nan)

# converting dataframe to an array of arrays to be suitable to use with the apyori library
voting_records = []
number_of_records = 0
for i in range(dataAssociation.shape[0]):
    voting_records.append(dataAssociation.iloc[i].dropna().tolist())
    number_of_records = i

# running apriori 
rules = apriori(voting_records, min_support=0.5, min_confidence=0.8)

# printing out generated rules with respective support and confidence levels
print("Rules:")
for rule in rules:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        '\nSupport:',rule.support, 'Confidence:', rule.ordered_statistics[0].confidence, 'Lift:', rule.ordered_statistics[0].lift, "\n")

    


