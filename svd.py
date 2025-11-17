# Singular Value Decomposition

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

