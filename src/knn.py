"""
CS6140 Project 2
Yihan Xu
Jake Van Meter
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from pca import pca, reduce_dimensions
from typing import *
from nn import plot_conf_matrix

ACTIVITY_LABELS = [
    1, # WALKING
    2, # WALKING_UPSTAIRS
    3, # WALKING_DOWNSTAIRS
    4, # SITTING
    5, # STANDING
    6, # LAYING
]

def load_uci_data() -> Tuple[pd.DataFrame, pd.Series]:
    # Load data from train and test sets
    X_names = np.loadtxt("datasets/UCI_HAR_Dataset/features.txt", usecols=1, dtype="str")
    y_name = "activity"
    test_X = np.loadtxt("datasets/UCI_HAR_Dataset/test/X_test.txt")
    test_y = np.loadtxt("datasets/UCI_HAR_Dataset/test/y_test.txt")
    train_X = np.loadtxt("datasets/UCI_HAR_Dataset/train/X_train.txt")
    train_y = np.loadtxt("datasets/UCI_HAR_Dataset/train/y_train.txt")
    # Re-form raw data set from loaded sets
    X_raw = np.concatenate((test_X, train_X), axis=0)
    y_raw = np.concatenate((test_y, train_y), axis=0)
    X_raw = pd.DataFrame(X_raw, columns=X_names)
    y_raw = pd.Series(y_raw, name=y_name)
    return X_raw, y_raw

# Finds the number of neighbors that produces the best score for the given training and test sets
def find_optimal_num_neighbors(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, \
                                range_start=2, range_end=11, print_all_scores=False) -> Tuple[int, float]:
    # Classify data set using KNN (Features are already whitened)
    max_score = 0.0
    best_n_neighbors = 0
    for i in range(range_start, range_end):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        if print_all_scores:
            print(f"Score for {i} neighbors: {score}")
        # Pick the number of neighbors that produces the highest score
        if score > max_score:
            max_score = score
            best_n_neighbors = i
    return best_n_neighbors, max_score

def build_and_fit_knn_classifier(num_neighbors: int, X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    ret_model = KNeighborsClassifier(n_neighbors=num_neighbors)
    ret_model.fit(X_train, y_train)
    return ret_model

def print_knn_results(knn: KNeighborsClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = knn.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Accuracy for k={knn.n_neighbors}: {knn.score(X_test, y_test)}")
    print(f"Confusion Matrix where k={knn.n_neighbors}:\n{conf_matrix}")
    plot_conf_matrix(conf_matrix)
    print("")

def main():
    X_raw, y_raw = load_uci_data()

    # Split data into 75% training data and 25% test data
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.25)

    # Classify data set using KNN (Features are already whitened)
    n_neigh, score, knn = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)

    # Execute PCA on data
    _, _, eigenvalues, _, proj_data = pca(X_raw, normalize=True)
    # Keep enough eigenvectors to explain 90% of the data
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.9)

    # Split the reduced-dimension data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)

    # Train the new classifier on the reduced-dimension data
    n_neigh, score, knn = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
