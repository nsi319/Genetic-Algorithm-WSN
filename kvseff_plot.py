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
    node_centroids = []

    for i in range(len(centroids)):
        dist=[]
        for j in range(N):
            x1 = data.iloc[j][0]
            y1 = data.iloc[j][1]
            x2 = centroids[i][0]
            y2 = centroids[i][1]
            dist.append((pow(x1-x2,2) + pow(y1-y2,2),j))
        dist = sorted(dist,key=lambda d: d[0])
        j = dist[0][1]
        x_c = data.iloc[j][0]
        y_c = data.iloc[j][1]
        node_centroids.append((x_c,y_c))
    
    d = 0
    for i in range(N):
        d = d + pow(data.iloc[i][0] - node_centroids[labels[i]][0],2) + pow(data.iloc[i][1]- node_centroids[labels[i]][1],2)
    
    node_centroids = list(set(node_centroids))
    for i in range(len(node_centroids)):
        d = d + pow(node_centroids[i][0],2) + pow(node_centroids[i][1],2)
    
    print("Max achieved Reduction in Energy (k-means): {:0.2f}%".format(100-(d/D)*100))
    return 100-(d/D)*100

k_value = []
eff = []
for i in range(2,8):
    k_value.append(i)
    eff.append(kmeans_plot(i))

best_k = eff.index(max(eff)) + 2
print("Best K: ",best_k)
plt.plot(k_value,eff)
plt.xlabel("k Value")
plt.ylabel("Reduction in Energy %")
plt.title("K vs Efficiency Plot")
plt.show()
kmeans_cluster(best_k,0)
