from __future__ import annotations

import pandas as pd
import numpy as np
import math
from typing import *
from utils import partition_training_and_test_data


class Column:
    def __init__(self, name: str, min: float, max: float, mean: float, median: float, variance: float, std: float, values: List) -> None:
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
    dist = 0
    # Convert the series' to lists for easy iteration
    first_vals = first.to_list()
    second_vals = second.to_list()
    for i in range(len(first_vals)):
        dist += (first_vals[i] - second_vals[i]) ** 2
    return math.sqrt(dist)


def normalized_dist(first: pd.Series, second: pd.Series, std: List[float]) -> float:
    dist = 0
    # Convert the series' to lists for easy iteration
    first_vals = first.to_list()
    second_vals = second.to_list()
    for i in range(len(first_vals)):
        dist += (first_vals[i] - second_vals[i]) ** 2
        dist /= std[i]  # Divide by the std for that col to whiten
    return math.sqrt(dist)


def dist_between_data_sets(from_points: pd.DataFrame, to_points: pd.DataFrame, std: List[float] | None) -> pd.DataFrame:
    # ExN distance matrix
    ret = []
    for i in range(from_points.shape[0]):
        from_data_point = from_points.iloc[i]
        col = []
        # Get the distance from the data point to each exemplar data point
        for j in range(to_points.shape[0]):
            to_data_point = to_points.iloc[j]
            dist = None
            if not std:
                dist = non_normalized_dist(from_data_point, to_data_point)
            else:
                dist = normalized_dist(from_data_point, to_data_point, std)
            col.append(dist)
        ret.append(col)
    return pd.DataFrame(ret)

def dist_between_data_points(from_point: pd.Series, to_points: pd.DataFrame, std: List[float] | None) -> pd.Series:
    # Ex1 distance matrix
    ret = []
    # Get the distance from the data point to each exemplar data point
    for i in range(to_points.shape[0]):
        to_data_point = to_points.iloc[i]
        dist = None
        if not std:
            dist = non_normalized_dist(from_point, to_data_point)
        else:
            dist = normalized_dist(from_point, to_data_point, std)
        ret.append(dist)
    return pd.Series(ret)

def nearest_neighbor(training_set: pd.DataFrame, test_set: pd.DataFrame, std: List[float]) -> Tuple[List[str], List[float]]:
    # Returned arrays
    predicted_classes = []
    error_terms = []

    # Get the distance matrix
    test_features = test_set.iloc[:, :-1]  # Assumes dependent variable is the last column
    training_features = training_set.iloc[:, :-1]
    dist_matrix = dist_between_data_sets(test_features, training_features, std[:-1])

    # Classify price based on distance to nearest neigbor
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

    # Run Nearest Neighbor
    std = [col.std for col in columns]
    predictions, error_terms = nearest_neighbor(training_df, test_df, std)

    # print(test_df["Price"])
    # print(predictions)

    # Histogram
    # df["Price"].plot(kind="hist", xlabel="Price (Million)")
    # plt.show()


if __name__ == "__main__":
    main()
