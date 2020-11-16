import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from scipy.spatial import distance
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage


def main():
    data = pd.read_csv('data/points.csv', header=None)
    print(min(data.iloc[:][0]),max(data.iloc[:][0]),min(data.iloc[:][1]),max(data.iloc[:][1]))
    clf = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    clf.fit_predict(data)
    print(clf.labels_)
    plt.scatter(data.iloc[:][0],data.iloc[:][1], c=clf.labels_.tolist(),cmap='rainbow')
    plt.show()

if __name__ == "__main__":
    main()
