import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift
import scipy.stats as stats
from typing import List
import math


# function to plot the clusters
def plot_cluster(data):
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.show()

def plot_cluster_with_label(label, data):
    # Getting unique labels
    u_labels = np.unique(label)
    # plotting the results:
    for i in u_labels:
        plt.scatter(data.iloc[label == i, 0], data.iloc[label == i, 1], label=i)
    plt.legend()
    plt.show()

# function to apply k-means clustering algorithm on a dataset
def k_means(data):
    kmeans = KMeans(n_clusters=6, n_init="auto")
    label = kmeans.fit_predict(data)
    print(label)
    plot_cluster_with_label(label, data)

def mean_shift(data):
    meanshift = MeanShift(bandwidth=2)
    label = meanshift.fit_predict(data)
    print(label)
    plot_cluster_with_label(label, data)


def main():
    # get the datasets
    df_1 = pd.read_csv("../datasets/clusterDataA-1.csv")
    df_2 = pd.read_csv("../datasets/clusterDataB-1.csv")
    # plot the datasets
    # plot_cluster(df_1)
    # plot_cluster(df_2)
    # apply k-means on the datasets
    # k_means(df_1)
    # k_means(df_2)
    # mean_shift(df_1)
    mean_shift(df_2)


if __name__ == "__main__":
    main()
