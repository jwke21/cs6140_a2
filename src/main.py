"""
CS6140 Project 2
Jake Van Meter
Yihan Xu
"""

from pca import *
from knn import *
from cluster import *


def main():
    #################### PART 1 ####################

    #################### PART 2 ####################

    #################### PART 3 ####################

    #################### PART 4 ####################
    X_raw, y_raw = load_uci_data()
    # Split data into 75% training data and 25% test data
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.25, random_state=11)

    print(f"\n-------------------- KNN ON RAW DATA --------------------\n")

    # Find optimal number of neighbors for training set
    n_neigh, score = find_optimal_num_neighbors(X_train, X_test, y_train, y_test, print_all_scores=True)
    print(f"{n_neigh} neighbors produced the highest accuracy of {score}")
    # Train the classifier
    knn = build_and_fit_knn_classifier(n_neigh, X_train, y_train)
    # Classify test set and print results
    print_knn_results(knn, X_test, y_test)

    print(f"\n-------------------- KNN ON REDUCED DATA SET --------------------\n")

    # Execute PCA on data
    _, _, eigenvalues, _, proj_data = pca(X_raw, normalize=True)
    # Keep enough eigenvectors to explain 90% of the data
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.9)
    # Split the reduced-dimension data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)
    # Train new classifier on reduced data set with same k
    knn = build_and_fit_knn_classifier(n_neigh, X_train, y_train)

    print_knn_results(knn, X_test, y_test)

    print(f"\n-------------------- KNN ON REDUCED DATA SET --------------------\n")

    # Keep enough eigenvectors to explain 80% of the data (Extension)
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)
    knn = build_and_fit_knn_classifier(n_neigh, X_train, y_train)

    print_knn_results(knn, X_test, y_test)

    print(f"\n-------------------- KNN ON REDUCED DATA SET --------------------\n")

    # Keep enough eigenvectors to explain 70% of the data (Extension)
    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.7)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_raw, test_size=0.25)
    knn = build_and_fit_knn_classifier(n_neigh, X_train, y_train)

    print_knn_results(knn, X_test, y_test)

    print(f"\n-------------------- K-MEANS ON REDUCED DATA SET --------------------\n")

    X_reduced = reduce_dimensions(proj_data, eigenvalues, explained_variance=0.9)

    # Use k-means to classify reduced data that explains 90% of variance
    rep_error = k_means(X_reduced, 6, False) # 6 clusters: one for each activity
    print(f"K-means representation error for k=6 on projected data: {rep_error}")

if __name__ == "__main__":
    main()
