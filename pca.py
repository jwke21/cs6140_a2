"""
CS6140 Project 2
Jake Van Meter
Yihan Xu
"""

import pandas as pd
import numpy as np

# computes the principal components of the given data
# returns the means, standard deviations, eigenvalues, eigenvectors, and projected data
def pca(data: pd.DataFrame, normalize=False, print_results=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    # assign to A the data as a numpy matrix
    A = data.to_numpy()
    # assign to m the mean values of the columns of A
    m = np.mean(A, axis=0) # 'axis=0' argument to return mean of each column in A
    # assign to D the difference matrix A - m
    D = A - m
    # Whiten
    if normalize:
        # Compute the standard deviations of each column
        std_deviations = np.std(A, axis=0)
    else:
        # Assign all 1s to the standard deviation vector (1 for each column)
        std_deviations = np.array([1 for _ in range(0, A.shape[1])])
    # Divide each column by its standard deviation vector
    #    (hint: this can be done as a single operation)
    D /= std_deviations
    # assign to U, S, V the result of running np.svd on D, with full_matrices=False
    U, S, V = np.linalg.svd(D, full_matrices=False)
    # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
    #   divided by the degrees of freedom (N-1). The values are sorted.
    N = A.shape[0]
    eigenvalues = S ** 2 / (N - 1)
    # project the data onto the eigenvectors. Treat V as a transformation 
    #   matrix and right-multiply it by D transpose. The eigenvectors of A 
    #   are the rows of V. The eigenvectors match the order of the eigenvalues.
    projected_data = np.dot(D, V.T)
    # create a new data frame out of the projected data
    new_df = pd.DataFrame(projected_data, columns=data.columns)
    # print the principal components
    if print_results:
        if not normalize:
            print("--------------------PCA results without whitening--------------------")
        else:
            print("--------------------PCA results with whitening--------------------")
        print("")
        print_pca_results(m, std_deviations, D, eigenvalues, V, projected_data, np.cov(D))
    # return the means, standard deviations, eigenvalues, eigenvectors, and projected dataD
    return m, std_deviations, eigenvalues, V, new_df

def reduce_dimensions(proj_data: pd.DataFrame, eigenvalues: np.ndarray, explained_variance: float) -> pd.DataFrame:
    N = eigenvalues.shape[0]
    variance = 0.0
    num_kept = 0
    # Find the number of required eigenvectors to capture explained variance
    for i, val in enumerate(eigenvalues):
        if variance / N > explained_variance:
            num_kept = i + 1
            break
        variance += val
    # Perform the dimensionality reduction
    X_reduced = proj_data.iloc[:, :num_kept]
    print(f"Number of eigenvectors kept: {num_kept} out of {N}")
    print(f"Explained variance: {variance / N}")

    return X_reduced

def print_pca_results(means: np.ndarray, std_deviations: np.ndarray,
                        difference_matrix: np.ndarray, eigenvalues: np.ndarray,
                        eigenvectors: np.ndarray, projected_data: pd.DataFrame,
                        covariance_matrix: np.ndarray) -> None:
    print(f"means: {means}")
    print(f"stdev: {std_deviations}\n")
    print(f"Difference matrix:\n{difference_matrix}\n")
    print(f"Eigenvalues:\n{eigenvalues}\n")
    print(f"Eigenvectors:\n{eigenvectors}\n")
    print(f"Projected data:\n{projected_data}\n")
    print(f"Covariance matrix:\n{covariance_matrix}\n")

