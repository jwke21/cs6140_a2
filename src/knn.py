import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pca import pca, reduce_dimensions
from typing import *

ACTIVITY_LABELS = [
    1, # WALKING
    2, # WALKING_UPSTAIRS
    3, # WALKING_DOWNSTAIRS
    4, # SITTING
    5, # STANDING
    6, # LAYING
]

# Finds the number of neighbors that produces the best score for the given training and test sets
def find_optimal_num_neighbors(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, \
                                range_start=2, range_end=11, print_all_scores=False) -> Tuple[int, float]:
    # Classify data set using KNN (Features are already whitened)
    max_score = 0.0
    best_n_neighbors = 0
    for i in range(range_end, range_end):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        if print_all_scores:
            print(f"Score for {i} neighbors: {score}")
        # Pick the number of neighbors that produces the highest score
        if score > max_score:
            max_score = score
            best_n_neighbors = i
    print(f"{best_n_neighbors} neighbors produced the highest score of {max_score}")
    return best_n_neighbors, max_score

def main():
    X_cols = np.loadtxt("datasets/UCI_HAR_Dataset/features.txt", usecols=1, dtype="str")
    y_col = "activity"

    # Reform raw data set before train-test split
    test_X = np.loadtxt("datasets/UCI_HAR_Dataset/test/X_test.txt")
    test_y = np.loadtxt("datasets/UCI_HAR_Dataset/test/y_test.txt")
    train_X = np.loadtxt("datasets/UCI_HAR_Dataset/train/X_train.txt")
    train_y = np.loadtxt("datasets/UCI_HAR_Dataset/train/y_train.txt")
    X_raw = np.concatenate((test_X, train_X), axis=0)
    y_raw = np.concatenate((test_y, train_y), axis=0)
    X_raw = pd.DataFrame(X_raw, columns=X_cols)
    y_raw = pd.Series(y_raw, name=y_col)

    # Re-split data into 75% training data and 25% test data
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.25)

    # Classify data set using KNN (Features are already whitened)
    n_neigh, score = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)
    knn = KNeighborsClassifier(n_neighbors=n_neigh)

    # Execute PCA on data
    _, _, eigenvalues, _, proj_data = pca(X_raw, normalize=True)
    # Keep enough eigenvectors to explain 90% of the data
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.9)

    # Split the reduced-dimension data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)

    # Train the new classifier on the reduced-dimension data
    n_neigh, score = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)
    knn = KNeighborsClassifier(n_neighbors=n_neigh)


if __name__ == "__main__":
    main()
