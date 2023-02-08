"""
CS6140 Project 2
Jake Van Meter
Yihan Xu
"""

from pca import *
from knn import *


def main():
    #################### PART 1 ####################

    #################### PART 2 ####################

    #################### PART 3 ####################

    #################### PART 4 ####################
    X_raw, y_raw = load_uci_data()
    # Split data into 75% training data and 25% test data
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.25)

    print(f"\n-------------------- KNN ON RAW DATA --------------------\n")

    # Classify data set using KNN (Features are already whitened)
    n_neigh, score, knn = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)

    # Print results of KNN on raw data
    print_knn_results(knn, X_test, y_test)

    print(f"\n-------------------- KNN ON REDUCED DATA SET --------------------\n")

    # Execute PCA on data
    _, _, eigenvalues, _, proj_data = pca(X_raw, normalize=True)
    # Keep enough eigenvectors to explain 90% of the data
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.9)
    # Split the reduced-dimension data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)
    # Train the new classifier on the reduced-dimension data
    n_neigh, score, knn = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)

    print_knn_results(knn, X_test, y_test)

    print(f"\n-------------------- KNN ON REDUCED DATA SET --------------------\n")

    # Keep enough eigenvectors to explain 80% of the data (Extension)
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)
    n_neigh, score, knn = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)

    print_knn_results(knn, X_test, y_test)

    print(f"\n-------------------- KNN ON REDUCED DATA SET --------------------\n")

    # Keep enough eigenvectors to explain 70% of the data (Extension)
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.7)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)
    n_neigh, score, knn = find_optimal_num_neighbors(X_train, X_test, y_train, y_test)

    print_knn_results(knn, X_test, y_test)



if __name__ == "__main__":
    main()
