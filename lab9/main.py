# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 1) (10 points) Load the data from the file 'dataOutliers.npy'
data = np.load("./dataOutliers.npy")

# 2) (10 points) Create a scatter plot to visualize the data (This is just a FYI, make sure to comment the below line after you visualized the data)
# not sure if the instruction means 'comment out' when it says 'comment', 
# but i decided not to comment out the lines, just using plt.show() to save and reset the plt 
plt.scatter(data[:,0], data[:,1])
plt.title(label="Scatter Plot")
plt.show()

# 3) (50 points) Anomaly detection: Density-based
# Fit the LocalOutlierFactor model for outlier detection
# Then predict the outlier detection labels of the data points
lof = LocalOutlierFactor()
prediction = lof.fit_predict(data)
outliers = data[np.where(prediction==-1)]

# 4) (30 points) Plot results: make sure all plots/images are closed before running the below commands
# Create a scatter plot of the data (exact same as in 2) )
# Then, indicate which points are outliers by plotting circles around the outliers
# clearing plt
plt.cla()
# plotting data
plt.scatter(data[:,0], data[:,1])
# plotting circles around outliers
plt.scatter(outliers[:,0], outliers[:,1], s=300, edgecolors="r", facecolors="none",label="Outlier")
plt.title(label="Scatter Plot with outliers highlighted")

