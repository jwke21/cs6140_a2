import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
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


def calculate_representation_error(kmeans_model, data, k):
    # get the squared distances
    distances = kmeans_model.transform(data) ** 2
    # find the distance to the closest cluster mean
    distances = distances.min(axis=1)
    # get the sum-squared error of those distances
    representation_error = 0
    for i in range(0, len(distances)):
        representation_error += distances[i]
    # verify if the representation error is calculated correctly
    if round(kmeans_model.inertia_, 3) != round(representation_error, 3):
        print(f"Representation error is not calculated correctly using k={k}")
    return representation_error


def calculate_minimum_description_length(k, n, representation_error):
    minimum_description_length = representation_error + (k / 2) * math.log2(n)
    return minimum_description_length

    # function to apply k-means clustering algorithm on a dataset


def k_means(data, k):
    kmeans = KMeans(n_clusters=k, n_init="auto")
    label = kmeans.fit_predict(data)
    # plot_cluster_with_label(label, data)
    representation_error = calculate_representation_error(kmeans, data, k)
    return representation_error


# function to apply mean-shift clustering algorithm on a dataset
def mean_shift(data):
    meanshift = MeanShift(bandwidth=2)
    label = meanshift.fit_predict(data)
    plot_cluster_with_label(label, data)


# function to apply hierarchical clustering algorithm on a dataset
def hierarchical_clustering(data):
    linkage_data = linkage(data, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    plt.show()
    hierarchical_cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
    label = hierarchical_cluster.fit_predict(data)
    plot_cluster_with_label(label, data)


def main():
    # get the datasets
    df_1 = pd.read_csv("../datasets/clusterDataA-1.csv")
    df_2 = pd.read_csv("../datasets/clusterDataB-1.csv")
    # plot the datasets
    # plot_cluster(df_1)
    # plot_cluster(df_2)

    # apply k-means on the datasets
    # k_means(df_1, 6)
    # k_means(df_2, 6)

    # apply mean shift on the datasets
    # mean_shift(df_1)
    # mean_shift(df_2)

    # apply hierarchical clustering on the datasets
    # hierarchical_clustering(df_1)
    # hierarchical_clustering(df_2)

    # try k from 2 to 10 on dataset A
    representation_errors = []
    k_values = []
    minimum_description_lengths = []
    n = len(df_1.iloc[:, 0])
    for i in range(2, 11):
        representation_error = k_means(df_1, i)
        representation_errors.append(representation_error)
        minimum_description_length = calculate_minimum_description_length(i, n, representation_error)
        minimum_description_lengths.append(minimum_description_length)
        k_values.append(i)
    plt.plot(k_values, representation_errors)
    plt.show()
    plt.plot(k_values, minimum_description_lengths)
    plt.show()


if __name__ == "__main__":
    main()
