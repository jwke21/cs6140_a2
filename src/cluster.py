import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import math


# Function to plot the clusters
def plot_cluster(data: pd.DataFrame) -> None:
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.show()


# Function to plot the clusters with labels
def plot_cluster_with_label(label: any, data: pd.DataFrame) -> None:
    # Getting unique labels
    u_labels = np.unique(label)
    # Plotting the results:
    for i in u_labels:
        plt.scatter(data.iloc[label == i, 0], data.iloc[label == i, 1], label=i)
    plt.legend()
    plt.show()


# Function to calculate representation error
def calculate_representation_error(kmeans_model: KMeans, data: pd.DataFrame, k: int) -> float:
    # Get the squared distances
    distances = kmeans_model.transform(data) ** 2
    # Find the distance to the closest cluster mean
    distances = distances.min(axis=1)
    # Get the sum-squared error of those distances
    representation_error = 0
    for i in range(0, len(distances)):
        representation_error += distances[i]
    # Verify if the representation error is calculated correctly
    if round(kmeans_model.inertia_, 3) != round(representation_error, 3):
        print(f"Representation error is not calculated correctly using k={k}")
    return representation_error


# Function to calculate minimum description length
def calculate_minimum_description_length(k: int, n: int, representation_error: float) -> float:
    minimum_description_length = representation_error + (k / 2) * math.log2(n)
    return minimum_description_length


def prepare_for_diff(k: int, p: int, data: pd.DataFrame) -> float:
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit_predict(data)
    representation_error = calculate_representation_error(kmeans, data, k)
    return (k ** (2 / p)) * representation_error


def calculate_diff(k: int, p: int, data: pd.DataFrame) -> float:
    diff = prepare_for_diff(k - 1, p, data) - prepare_for_diff(k, p, data)
    return diff


def calculate_KL(k: int, data: pd.DataFrame) -> float:
    p = len(data.axes[1])
    diff_k = calculate_diff(k, p, data)
    diff_k_plus = calculate_diff(k + 1, p, data)
    return abs(diff_k / diff_k_plus)


# Function to generate clusters with k means algorithm on a dataset with input k
def k_means(data: pd.DataFrame, k: int, if_plot: bool = True) -> float:
    kmeans = KMeans(n_clusters=k, n_init="auto")
    label = kmeans.fit_predict(data)
    if if_plot:
        plot_cluster_with_label(label, data)
    representation_error = calculate_representation_error(kmeans, data, k)
    return representation_error


# Function to generate clusters with mean shift algorithm on a dataset
def mean_shift(data: pd.DataFrame) -> None:
    meanshift = MeanShift(bandwidth=2)
    label = meanshift.fit_predict(data)
    plot_cluster_with_label(label, data)


# Function to generate clusters with hierarchical algorithm on a dataset
def hierarchical_clustering(data: pd.DataFrame) -> None:
    linkage_data = linkage(data, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    plt.show()
    hierarchical_cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
    label = hierarchical_cluster.fit_predict(data)
    plot_cluster_with_label(label, data)


# Try k means clustering with a range of K and plot the quality metrics
def k_means_with_range_and_plot(min: int, max: int, data: pd.DataFrame, if_plot: bool = True) -> None:
    representation_errors = []
    k_values = []
    minimum_description_lengths = []
    KLs = []
    n = len(data.iloc[:, 0])
    for i in range(min, max):
        representation_error = k_means(data, i, if_plot)
        representation_errors.append(representation_error)
        minimum_description_length = calculate_minimum_description_length(i, n, representation_error)
        minimum_description_lengths.append(minimum_description_length)
        KL = calculate_KL(i, data)
        KLs.append(KL)
        k_values.append(i)
    plt.plot(k_values, representation_errors)
    plt.show()
    plt.plot(k_values, minimum_description_lengths)
    plt.show()
    plt.plot(k_values, KLs)
    plt.show()


def main():
    # Get the datasets
    df_1 = pd.read_csv("../datasets/clusterDataA-1.csv")
    df_2 = pd.read_csv("../datasets/clusterDataB-1.csv")
    # Plot the datasets
    plot_cluster(df_1)
    plot_cluster(df_2)

    # Apply k-means on the datasets
    k_means(df_1, 6)
    k_means(df_2, 6)

    # Apply mean shift on the datasets
    mean_shift(df_1)
    mean_shift(df_2)

    # Apply hierarchical clustering on the datasets
    hierarchical_clustering(df_1)
    hierarchical_clustering(df_2)

    # Try k from 2 to 10 on dataset A
    k_means_with_range_and_plot(2, 11, df_1)

    # Apply K-Means Clustering to wine dataset
    wine_data = pd.read_csv("../datasets/wine-clustering.csv")
    # Try k from 2 to 30 on dataset A
    k_means_with_range_and_plot(2, 30, wine_data, False)


if __name__ == "__main__":
    main()
