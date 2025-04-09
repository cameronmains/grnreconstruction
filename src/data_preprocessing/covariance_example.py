import numpy as np

X1 = np.array([
    [2, 3, 5],
    [3, 2, 6],
    [2, 4, 5],
    [3, 3, 6]
])

# Class w2 data (example; you can replace with your own points)
X2 = np.array([
    [6, 8, 9],
    [7, 7, 10],
    [6, 9, 8],
    [7, 8, 10]
])

# Compute covariance matrices
cov_w1 = np.cov(X1, rowvar=False)
cov_w2 = np.cov(X2, rowvar=False)

# Print results
print("Covariance matrix for class w1:\n", cov_w1)
print("\nCovariance matrix for class w2:\n", cov_w2)