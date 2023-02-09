"""
CS6140 Project 2
Jake Van Meter
Yihan Xu
"""

import pandas as pd
import numpy as np
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
    """ Euclidian Distance Between Data sets calculation.

    Calculates the Eculidian distance from the point(s) in one set to the points in the other
    set. Distance is non-normalized by default but can be normalized if an ordered list
    of standard deviations for each feature is given. Given an NxF data set and an
    ExF data set, will return a ExN data set whose rows are the euclidian distance from
    each of the N samples of the first matrix (now columns) to to each of the E samples
    of second matrix (now rows).

    Parameters
    ----------
    from_points : pandas.Dataframe or pandas.Series
        The NxF matrix whose N row(s) will be the columns of the new matrix.

    to_points : pandas.Dataframe
        The ExF matrix whose E rows will be the rows of the new matrix.

    std : List[float] or None, default=None
        The list of standard deviations for each of the F features in the overall data set.
        Used for getting normalized distances between data points. If None, non-
        normalized Euclidian distance will be used. Otherwise, the values of the
        list will be used to get the normalized Euclidian distance.

    Returns
    -------
    ret : pandas.DataFrame or pandas.Series
        The ExN matrix whose rows are the Euclidian distance from each of the N from_points
        samples to each of the E to_points samples. 

    """
    # ret will be ExN distance matrix using a standardized Euclidian distance metric
    ret = None

    if not std:
        std = [1 for _ in range(len(from_points.columns))]

    # Handle case where N=1: from_points will be a Series
    if len(from_points.shape) == 1:
        # Transform the from_points into a single row
        transformed_from_points = np.array(from_points).reshape(1, -1)
        # Get the square root of the sum of squares divided by std (1 if not normalized)
        ret = (transformed_from_points - to_points.values)**2/std
        ret = np.sum(ret, axis=-1)
        ret = np.sqrt(ret)
        ret = pd.Series(ret)
    else:
        # Transform the from_points and to_points into a 1xNxF matrix and Ex1xF matrix respectively
        # (where F is number of features) so that numpy's broadcasting will transform their summ
        # squared differened along the F features into a ExN matrix
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        # https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis
        transformed_from_points = from_points.values[np.newaxis, :, :] # np.newaxis adds dimension
        transformed_to_points = to_points.values[:, np.newaxis, :]
        # Get the square root of the sum of squares divided by std (1 if not normalized) of the distance between the points
        ret = (transformed_from_points - transformed_to_points)**2/std
        ret = np.sum(ret, axis=-1)
        ret = np.sqrt(ret)
        ret = pd.DataFrame(ret)

    return ret


def k_nearest_neighbors(training_set: pd.DataFrame, test_set: pd.DataFrame, k: int, std: List[float] | None = None) -> Tuple[List[int], List[float]]:
    """ K-Nearest Neighbor (KNN) implementation.

    Utilizes a Euclidian Distance metric to classify test data points based on the labels
    of the labels of their k closest neigbors. Distance metric is non-normalized by default
    but can be normalized if an ordered list of standard deviations for each feature is given.

    Parameters
    ----------
    training_set : pandas.DataFrame
        The training data that will be used to train the model.

    test_set : pandas.DataFrame
        The test set whose dependent values will be predcited from.

    k : int
        The number of neighbors that will be used to classify the test data points.

    std : List[float] or None, default=None
        The list of standard deviations for each feature in the overall data set.
        Used for getting normalized distances between data points. If None, the
        difference metric will not be normalized. None by default.

    Returns
    -------
    predicted_classes : List[int]
        The predicted classes for the data points in the test_set.

    error_terms : List[float]
        The error terms for the data points of the test set. A data point's
        error_term is the distance between it and its closest neighbor.
    """
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
        predicted_classes.append(Counter(classes).most_common(1)[0][0])
        closest_dist = distances_to_points.min()
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


def first_task():
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


if __name__ == "__main__":
    first_task()
