# Singular Value Decomposition
"""
1. Definition
SVD is a way to factorize any matrix A into three matrices ie A = U * Σ * V^T
Where:
- A      : original m x n matrix
- U      : m x m orthogonal matrix (left singular vectors)
- Σ      : m x n diagonal matrix (singular values)
- V^T    : n x n orthogonal matrix (transpose of right singular vectors)

2. Properties
- U and V^T are orthogonal: U^T * U = I, V^T * V = I
- Singular values in Σ are non-negative and usually sorted in descending order
- Works for square and rectangular matrices

3. Why it is useful
- Dimensionality reduction (keep top k singular values)
- Noise removal and data compression
- Solving linear systems, especially non-square or singular matrices
- Image compression and Latent Semantic Analysis (text analysis)
"""
# Example in Python
# Import required libraries
import numpy as np
from numpy.linalg import svd

    # For a 2 by 2 matrix
# Define the matrix
A = np.array([[3, 1],
              [1, 3]])

# SVD
U, singular, Vt = svd(A)

# Build Sigma (2x2)
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, singular)

print("A is:\n", A)
print("\nU is:\n", U)
print("\nSingular values are:", singular)
print("\nSigma is:\n", Sigma)
print("\nV^T is:\n", Vt)

    # For a 3 by 3 matrix
# Define the matrix
B = np.array([[4, 2, 1],
              [0, 3, -1],
              [2, 1, 5]])

# SVD
U, singular, Vt = svd(B)

# Build Sigma (3x3)
Sigma = np.zeros_like(B, dtype=float)
np.fill_diagonal(Sigma, singular)

print("B is:\n", B)
print("\nU is:\n", U)
print("\nSingular values are:", singular)
print("\nSigma is:\n", Sigma)
print("\nV^T is:\n", Vt)

    # For a 2 by 3 matrix
# Define matrix
A = np.array([[3, 2, 2], 
              [2, 3, -2]])

# SVD
U, singular, Vt = svd(A)

# Build Sigma
Sigma = np.zeros_like(A,dtype = float)
np.fill_diagonal(Sigma, singular)

print("B is:\n", A)
print("\nU is:\n", U)
print("\nSingular values are:", singular)
print("\nSigma is:\n", Sigma)
print("\nV^T is:\n", Vt)


