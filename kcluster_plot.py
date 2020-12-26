import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial import distance
import pandas as pd
import math


def kmeans_cluster(cluster):
    data = pd.read_csv('data/points.csv', header=None)
    D = 0
    N = data.shape[0]
    x = []
    y = []
    for i in range(N):
        D = D + pow(data.iloc[i][0],2) + pow(data.iloc[i][1],2)
        x.append(data.iloc[i][0])
        y.append(data.iloc[i][1])

    
    clf = KMeans(n_clusters=cluster)
    clf.fit_predict(data)
    labels = clf.labels_.tolist()
    centroids = clf.cluster_centers_.tolist()
    d = 0
    for i in range(N):
        d = d + pow(data.iloc[i][0] - centroids[labels[i]][0],2) + pow(data.iloc[i][1]- centroids[labels[i]][1],2)
    for i in range(len(centroids)):
        d = d + pow(centroids[i][0],2) + pow(centroids[i][1],2)
    
    print("Energy reduction (k-means): {:0.2f}%".format(100-(d/D)*100))

    plt.scatter(x,y, c=clf.predict(data), s=50, cmap='viridis')
    centers = clf.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.show()
    
kmeans_cluster(5)