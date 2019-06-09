import numpy as np

# Square matrix:
A = np.array([[1, 2], [3, 4]])

print("A:\n", A)
print("A rows:\n", A[0], A[1])
print("A transpose:\n", A.T) # A transpose
print("A cols:\n", A.T[0], A.T[1])

# Matrix multiply with @
Q = A.T @ A
print("(A^T)A:\n", Q) # expect [[10, 14], [14, 20]]


# Non-square matrix
B = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 matrix
print("B:\n", B, B.shape)
print("(B^T)B:\n", B.T @ B, (B.T @ B).shape) # expect 3x3 [[17, 22, 27], [22, 29, 36], [27, 36, 45]]
print("B(B^T):\n", B @ B.T, (B @ B.T).shape) # expect 2x2 [[14, 32], [32, 77]]
print()

# Matrix inversion:
Qinv = np.linalg.inv(Q)
print("Q = (A^T)A:\n", Q)
print("Q^-1:\n", Qinv)
print("(Q^-1)Q:\n", Qinv @ Q)
print("Q(Q^-1):\n", Q @ Qinv)

# eigendecomposition (eigenvector/value v,lambda: Av = lambda*v)
(d, V) =  np.linalg.eig(Q) # d is eigenvalues, V is matrix whose columns are eigenvectors

# eigenvectors are orthnormal - columns are length 1, dot-products of columns is 0
print("\nEigenvectors orthonormal check:")
print(np.linalg.norm(V.T[0]), np.linalg.norm(V.T[1]), np.dot(V.T[0], V.T[1])) 

# verify d,V is indeed the eigendecomposition of Q:
print("\nEigendecomposition check:")
print("1st:", Q @ V.T[0], V.T[0]*d[0], np.allclose(Q @ V.T[0], V.T[0]*d[0]))
print("2nd:", Q @ V.T[1], V.T[1]*d[1], np.allclose(Q @ V.T[1], V.T[1]*d[1]))