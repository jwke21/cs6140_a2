import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import List
import math

class Column:
    def __init__(self, name: str, min: float, max: float, mean: float, \
            median: float, variance: float, std: float, values: List) -> None:
        self.name = name
        self.min = min
        self.max = max
        self.mean = mean
        self.median = median
        self.variance = variance
        self.std = std
        self.values = values
    
    def __str__(self) -> str:
        ret = "Column: {}\n".format(self.name) +\
        "Min: {}\n".format(self.min) +\
        "Max: {}\n".format(self.max) +\
        "Mean: {}\n".format(self.mean) +\
        "Median: {}\n".format(self.median) +\
        "Variance: {}\n".format(self.variance) +\
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
    return math.sqrt( dist )

def normalized_dist(first: pd.Series, second: pd.Series, std: pd.Series) -> float:
    dist = 0
    # Convert the series' to lists for easy iteration
    first_vals = first.to_list()
    second_vals = second.to_list()
    std_vals = std.to_list()
    for i in range(len(first_vals)):
        dist += (first_vals[i] - second_vals[i]) ** 2
        dist /= std_vals[i]  # Divide by the std for that col to whiten
    return math.sqrt( dist )

def distance_between_data_points(from_points: pd.DataFrame, to_points: pd.DataFrame, std: pd.Series | None) -> pd.DataFrame:
    pass

def main():
    df = pd.read_csv("USA_Housing3.csv")
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