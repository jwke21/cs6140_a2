"""
CS6140 Project 2
Yihan Xu
Jake Van Meter
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import math
from typing import *
from utils import partition_training_and_test_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Column:
    def __init__(self, name: str, min: float, max: float, mean: float, median: float, variance: float, std: float,
                 values: List) -> None:
        self.name = name
        self.min = min
        self.max = max
        self.mean = mean
        self.median = median
        self.variance = variance
        self.std = std
        self.values = values

    def __str__(self) -> str:
        ret = "Column: {}\n".format(self.name) + \
              "Min: {}\n".format(self.min) + \
              "Max: {}\n".format(self.max) + \
              "Mean: {}\n".format(self.mean) + \
              "Median: {}\n".format(self.median) + \
              "Variance: {}\n".format(self.variance) + \
              "Standard Deviation: {}\n".format(self.std)
        return ret


def build_column_from_series(series: pd.Series) -> Column:
    return Column(
        name=series.name,
        min=series.min(),
        max=series.max(),
        mean=series.mean(),
        median=series.median(),
        variance=series.var(),
        std=series.std(),
        values=series.values
    )


def dist_between_data_sets(from_points: pd.DataFrame | pd.Series, to_points: pd.DataFrame, std: List[float] | None) -> pd.DataFrame | pd.Series:
    # ret will be ExN distance matrix using a standardized Euclidian distance metric
    ret = None

    if not std:
        std = [1 for _ in range(len(from_points.columns))]

    # Handle case where N=1
    if from_points.shape[0] == 1:
        # Transform the from_points into a single column
        transformed_from_points = np.array(from_points).reshape(1, -1)
        # Get the square root of the sum of squares divided by std (1 if not normalized)
        ret = (transformed_from_points - to_points.values)**2/std
        ret = np.sum(ret, axis=-1)
        ret = np.sqrt(ret)
        ret = pd.Series(ret)
    else:
        # Transform the from_points and to_points into a 1xNx4 matrix and Ex1x4 matrix respectively 
        transformed_from_points = from_points.values[np.newaxis, :, :]
        transformed_to_points = to_points.values[:, np.newaxis, :]
        # Get the square root of the sum of squares divided by std (1 if not normalized) of the distance between the points
        ret = (transformed_from_points - transformed_to_points)**2/std
        ret = np.sum(ret, axis=-1)
        ret = np.sqrt(ret)
        ret = pd.DataFrame(ret)

    return ret


def k_nearest_neighbors(training_set: pd.DataFrame, test_set: pd.DataFrame, k: int, std: List[float] | None = None) -> Tuple[List[str], List[float]]:
    # Returned arrays
    predicted_classes = []
    error_terms = []

    # Get the distance matrix
    test_features = test_set.iloc[:, :-1]  # Assumes dependent variable is the last column
    training_features = training_set.iloc[:, :-1]
    if std:
        dist_matrix = dist_between_data_sets(test_features, training_features, std[:-1])
    else:
        dist_matrix = dist_between_data_sets(test_features, training_features, None)

    # Classify price based on distance to nearest neighbor
    for i in range(test_set.shape[0]):
        # Get distances
        distances_to_points = dist_matrix[i]
        # Get k nearest neighbors
        nearest_neighbors = distances_to_points.argsort()[:k]
        classes = [training_set["Price"].iloc[index] for index in nearest_neighbors]
        print(f"classes: {classes}")
        predicted_classes.append(Counter(classes).most_common(1)[0][0])
        closest_dist = distances_to_points.min()
        # nn_idx = distances_to_points[distances_to_points == closest_dist].index[0]
        # nn = training_set.iloc[nn_idx]
        # # Get nearest neighbor's class
        # nn_class = nn["Price"]
        # # Add predicted value to prediction array
        # predicted_classes.append(nn_class)
        # Add distance to error terms array
        error_terms.append(closest_dist)

    return (predicted_classes, error_terms)


def calculate_precision(actual_df: pd.DataFrame, predicted_df: List[str]) -> float:
    match = 0
    n = actual_df.shape[0]
    for i in range(0, n):
        if float(actual_df.iloc[i]) == float(predicted_df[i]):
            match += 1
    return match / n


def compute_confusion_matrix(actual_y: pd.DataFrame, predicted_y: pd.DataFrame) -> Any:
    conf_matrix = confusion_matrix(actual_y, predicted_y)
    return conf_matrix


def plot_conf_matrix(conf_matrix: Any) -> None:
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    df = pd.read_csv("datasets/USA_Housing3.csv")
    columns = []

    for i in range(len(df.columns.values)):
        col_name = df.columns.values[i]
        col = build_column_from_series(df[col_name])
        columns.append(col)
        print(col)

    # Price categories
    LOW = 750_000
    MED_LOW = 1_250_000
    MED_HI = 1_750_000
    # Turn price column into categorical data
    price_col = df["Price"]
    for i, val in enumerate(price_col):
        if val <= LOW:
            price_col[i] = 0
        elif val <= MED_LOW:
            price_col[i] = 1
        elif val <= MED_HI:
            price_col[i] = 2
        else:
            price_col[i] = 3

    # Split into test and training set
    test_df, training_df = partition_training_and_test_data(data=df, test_data_percentage=25)

    # Run Nearest Neighbor with whitened data
    std = [col.std for col in columns]
    predictions, error_terms = k_nearest_neighbors(training_df, test_df, 1, std)
    # Calculate precision
    precision = calculate_precision(test_df["Price"], predictions)
    print(f"precision using whitened data is: {precision}")
    # Calculate and plot confusion matrix
    conf_matrix = compute_confusion_matrix(test_df["Price"], predictions)
    plot_conf_matrix(conf_matrix)

    # Run Nearest Neighbor without whitened data
    predictions, error_terms = k_nearest_neighbors(training_df, test_df, 1)
    # Calculate precision
    precision = calculate_precision(test_df["Price"], predictions)
    print(f"precision using un_whitened data is: {precision}")
    # Calculate and plot confusion matrix
    conf_matrix = compute_confusion_matrix(test_df["Price"], predictions)
    plot_conf_matrix(conf_matrix)

    # Try different number of neighbors
    predictions, error_terms = k_nearest_neighbors(training_df, test_df, 10)
    # Calculate precision
    precision = calculate_precision(test_df["Price"], predictions)
    print(f"precision using un_whitened data with 10 nearest neighbors is: {precision}")
    # Calculate and plot confusion matrix
    conf_matrix = compute_confusion_matrix(test_df["Price"], predictions)
    plot_conf_matrix(conf_matrix)

    # Try different number of neighbors
    std = [col.std for col in columns]
    predictions, error_terms = k_nearest_neighbors(training_df, test_df, 10, std)
    # Calculate precision
    precision = calculate_precision(test_df["Price"], predictions)
    print(f"precision using whitened data with 10 nearest neighbors is: {precision}")
    # Calculate and plot confusion matrix
    conf_matrix = compute_confusion_matrix(test_df["Price"], predictions)
    plot_conf_matrix(conf_matrix)

    # Histogram
    # df["Price"].plot(kind="hist", xlabel="Price (Million)")
    # plt.show()


if __name__ == "__main__":
    main()
