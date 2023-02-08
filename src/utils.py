"""
CS6140 Project 2
Yihan Xu
Jake Van Meter
"""

import numpy as np
import pandas as pd
import random
import math
from matplotlib import pyplot as plt
from typing import *


def open_csv_as_df(path: str) -> pd.DataFrame:
    print(f"fetching csv from {path}")
    return pd.read_csv(path)


def partition_training_and_test_data(data: pd.DataFrame, test_data_percentage: float, usr_seed=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = len(data.index)
    test_data_percentage /= 100
    features = data.columns.values

    random.seed(usr_seed)
    added_indices = set()

                    ########## Construct test set ##########
    # Number of rows in test set = total_rows * given_percentage
    test_df_size = math.floor(total_rows * (test_data_percentage))
    test_df_indicies = []
    num_added = 0
    test_df = pd.DataFrame(columns=features)
    # Add random rows to test set until it is correct size
    print(f"\nbuilding test set of size {test_df_size}...")
    while num_added < test_df_size:
        row_index = random.randint(0, total_rows - 1)
        # Skip indices that have already been added to test set
        if row_index in added_indices:
            continue
        # Add the row to the test set
        added_indices.add(row_index)
        test_df_indicies.append(row_index)
        num_added += 1
    rows = pd.DataFrame(data.iloc[test_df_indicies], columns=features)
    test_df = pd.concat([test_df, rows])
    test_df.sort_index(inplace=True)
    print("\n--------------------TEST SET--------------------\n")
    print(f"Test set sample size: {test_df_size}")
    print(f"Test set sample percentage: {test_data_percentage}")

                    ########## Construct training set ##########
    training_df_indices = []
    num_added = 0
    training_df = pd.DataFrame(columns=features)
    # Training set will be composed of rows not added to test set
    for row_index in range(0, total_rows):
        # Skip indices that have already been added to training set or test set
        if row_index in added_indices:
            continue
        # Add row to training set
        added_indices.add(row_index)
        training_df_indices.append(row_index)
        num_added += 1
    rows = pd.DataFrame(data.iloc[training_df_indices], columns=features)
    training_df = pd.concat([training_df, rows])
    training_df.sort_index(inplace=True)
    print("\n--------------------TRAINING SET--------------------\n")
    print(f"Training set sample size: {total_rows - test_df_size}")
    print(f"Training set sample percentage: {1 - test_data_percentage}")
    print("\n")
    return [test_df, training_df]


def plot_df_scatter(data: pd.DataFrame, ind_var: str, dep_var: str) -> None:
    data.plot(x=ind_var, y=dep_var, kind="scatter")
    plt.xlabel(ind_var)
    plt.ylabel(dep_var)
    plt.show()
