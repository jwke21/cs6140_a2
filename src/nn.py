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


def non_normalized_dist(first: pd.Series, second: pd.Series) -> float:
    dist = np.square(first.values - second.values)
    dist = np.sum(dist)
    dist = np.sqrt(dist)
    return dist


def normalized_dist(first: pd.Series, second: pd.Series, std: List[float]) -> float:
    dist = np.square(first.values - second.values)
    dist = np.divide(dist, std)
    dist = np.sum(dist)
    dist = np.sqrt(dist)
    return dist


def dist_between_data_sets(from_points: pd.DataFrame, to_points: pd.DataFrame, std: List[float] | None) -> pd.DataFrame:
    # ExN distance matrix
    ret = pd.DataFrame()
    cols = []
    for i, from_data_point in from_points.iterrows():
        cols.append(dist_between_data_points(from_data_point, to_points, std))
    return pd.concat(cols, axis=1)


def dist_between_data_points(from_point: pd.Series, to_points: pd.DataFrame, std: List[float] | None) -> pd.Series:
    # Ex1 distance matrix
    ret = []
    # Get the distance from the data point to each exemplar data point
    for i, to_data_point in to_points.iterrows():
        dist = None
        if not std:
            dist = non_normalized_dist(from_point, to_data_point)
        else:
            dist = normalized_dist(from_point, to_data_point, std)
        ret.append(dist)
    return pd.Series(ret)


def nearest_neighbor(training_set: pd.DataFrame, test_set: pd.DataFrame, std: List[float] | None = None) -> Tuple[List[str], List[float]]:
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
        # Get nearest neighbor
        closest_dist = distances_to_points.min()
        nn_idx = distances_to_points[distances_to_points == closest_dist].index[0]
        nn = training_set.iloc[nn_idx]
        # Get nearest neighbor's class
        nn_class = nn["Price"]
        # Add predicted value to prediction array
        predicted_classes.append(nn_class)
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
    df = pd.read_csv("../datasets/USA_Housing3.csv")
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
    predictions, error_terms = nearest_neighbor(training_df, test_df, std)

    # Calculate precision
    precision = calculate_precision(test_df["Price"], predictions)
    print(f"precision using whitened data is: {precision}")
    # Calculate and plot confusion matrix
    conf_matrix = compute_confusion_matrix(test_df["Price"], predictions)
    plot_conf_matrix(conf_matrix)

    # Run Nearest Neighbor without whitened data
    predictions, error_terms = nearest_neighbor(training_df, test_df)

    # Calculate precision$
    precision = calculate_precision(test_df["Price"], predictions)
    print(f"precision using un_whitened data is: {precision}")
    # Calculate and plot confusion matrix
    conf_matrix = compute_confusion_matrix(test_df["Price"], predictions)
    plot_conf_matrix(conf_matrix)

    # Histogram
    # df["Price"].plot(kind="hist", xlabel="Price (Million)")
    # plt.show()


if __name__ == "__main__":
    main()
