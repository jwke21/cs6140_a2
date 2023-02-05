from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import List
import math


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


def distance_between_data_points(from_points: pd.DataFrame, to_points: pd.DataFrame, \
                                 std: List[float] | None) -> pd.DataFrame:
    # ExN distance matrix
    ret = pd.DataFrame()
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
        ret[i] = pd.Series(col)
    return ret

def dist_between_data_points(from_point: pd.Series, to_points: pd.DataFrame, std: List[float] | None) -> pd.Series:
    ret = None
    if not std:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
        ret = to_points.apply(
            function=non_normalized_dist, # "Function to apply to each column or row"
            axis=1, # "Apply function to each row"
            raw=False, # "Passes each row or column as a Series to the function"
            result_type="reduce", # "Returns a Series if possible"
            args=(from_point, this),
        )
    else:
        ret = to_points.apply(
            function=non_normalized_dist, # "Function to apply to each column or row"
            axis=1, # "Apply function to each row"
            raw=False, # "Passes each row or column as a Series to the function"
            result_type="reduce", # "Returns a Series if possible"
            args=(from_point, this, std),
        )
    return ret


def main():
    df = pd.read_csv("../datasets/USA_Housing3.csv")
    columns = []

    for i in range(len(df.columns.values)):
        col_name = df.columns.values[i]
        col = build_column_from_series(df[col_name])
        columns.append(col)
        print(col)

    # Histogram
    # df["Price"].plot(kind="hist", xlabel="Price (Million)")
    # plt.show()


if __name__ == "__main__":
    main()
