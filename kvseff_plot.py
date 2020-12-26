import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial import distance
import pandas as pd
import math

from kmeans import kmeans_cluster


def kmeans_plot(cluster):
    data = pd.read_csv('data/points.csv', header=None)
    D = 0
    N = data.shape[0]
    for i in range(N):
        D = D + pow(data.iloc[i][0],2) + pow(data.iloc[i][1],2)
    
    clf = KMeans(n_clusters=cluster)
    clf.fit_predict(data)
    labels = clf.labels_.tolist()
    centroids = clf.cluster_centers_.tolist()
    d = 0
    for i in range(N):
        d = d + pow(data.iloc[i][0] - centroids[labels[i]][0],2) + pow(data.iloc[i][1]- centroids[labels[i]][1],2)
    for i in range(cluster):
        d = d + pow(centroids[i][0],2) + pow(centroids[i][1],2)
    
    print("Energy reduction (k-means): {:0.2f}%".format(100-(d/D)*100))
    return 100-(d/D)*100

k_value = []
eff = []
for i in range(2,8):
    k_value.append(i)
    eff.append(kmeans_plot(i))

best_k = eff.index(max(eff)) + 2
print(best_k)
plt.plot(k_value,eff)
plt.xlabel("k Value")
plt.ylabel("Reduction in Energy %")
plt.show()
kmeans_cluster(best_k,0)
