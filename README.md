# Practicals-MFC-
# 1.	Create and transform vectors and matrices (the transpose vector (matrix) conjugate
# a.	transpose of a vector (matrix))

import numpy as np
nr=int(input('Enter the no of rows'))
nc=int(input('Enter no of columns'))
print('Enter entries in line separated by space')
entry=list(map(int, input().split()))
m=np.array(entry).reshape(nr,nc)
print('Matrix is', m)
print('Transpose is',np.transpose(m))

# 2.	Generate the matrix into echelon form and find its rank.
import numpy as np
nr=int(input('Enter the no of rows'))
nc=int(input('Enter no of columns'))
print('Enter entries in line separated by space')
entry=list(map(int, input().split()))
m=np.array(entry).reshape(nr,nc)
print('Matrix is', m)
print('Rank is', np.linalg.matrix_rank(m))

# 3.	Find cofactors, determinant, adjoint and inverse of a matrix.
import numpy as np
nr=int(input('Enter the no of rows'))
nc=int(input('Enter no of columns'))
print('Enter entries in line separated by space')
entry=list(map(int, input().split()))
m=np.array(entry).reshape(nr,nc)
print('Matrix is', m)
invr=np.linalg.inv(m)
trans=np.transpose(invr)
deter=np.linalg.det(m)
cofa=np.dot(trans,deter)
print('cofactor is', cofa)
print('determinant is', deter)
adjo=np.transpose(cofa)
print('Adjoint is', adjo) 

# 4.	Solve a system of Homogeneous and non-homogeneous equations using Gauss
# a.	elimination method.
import numpy as np
print("Enter the dimension of coefficientmatrix(A):")
NR = int(input("Enter the number of rows :"))
NC =int(input("Enter the number of columns:"))
print("Enter the elements of coefficients of matrix(A) in single line(seperated by space):")
Coefficients_Entries = list(map(float,input().split()))
Coefficients_Matrix = np.array(Coefficients_Entries).reshape(NR,NC)
print("Coefficient Matrix(A)is as follows:",'\n',Coefficients_Matrix,"\n")

# 5.	Solve a system of Homogeneous equations using the Gauss Jordan method.
import numpy as np
print("Enter the dimension of coefficientmatrix(A):")
NR = int(input("Enter the number of rows :"))
NC =int(input("Enter the number of columns:"))
print("Enter the elements of coefficients of matrix(A) in single line(seperated by space):")
Coefficients_Entries = list(map(float,input().split()))
Coefficients_Matrix = np.array(Coefficients_Entries).reshape(NR,NC)
print("Coefficient Matrix(A)is as follows:",'\n',Coefficients_Matrix,"\n")

# 6.	Generate basis of column space, null space, row space and left null space of a matrixspace.
import numpy as np

def get_column_space(A, tol=1e-10):
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > tol)
      return U[:, :rank]

def get_row_space(A, tol=1e-10):
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > tol)
    return Vt[:rank, :]

def get_null_space(A, tol=1e-10):
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > tol)
    return Vt[rank:, :].T

def get_left_null_space(A, tol=1e-10):
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > tol)
    return U[:, rank:]


m = int(input("Enter number of rows: "))
n = int(input("Enter number of columns: "))

A = np.zeros((m, n))
print("\nEnter matrix elements row by row:")
for i in range(m):
    A[i] = list(map(float, input(f"Row {i+1}: ").split()))

print("\nMatrix A:\n", A)


print("\n--- Basis of Column Space ---")
print(get_column_space(A))

print("\n--- Basis of Row Space ---")
print(get_row_space(A))

print("\n--- Basis of Null Space ---")
print(get_null_space(A))

print("\n--- Basis of Left Null Space ---")
print(get_left_null_space(A))

# 7.	Check the linear dependence of vectors. Generate a linear combination of given vectors of Rn/ matrices of the same size and find the transition matrix of given matrix space.
import numpy as np

def check_linear_dependence(vectors):
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    if rank < A.shape[1]:
        return False, rank
    return True, rank

def linear_combination(vectors, coefficients):
    result = np.zeros_like(vectors[0], dtype=float)
    for c, v in zip(coefficients, vectors):
        result += c * v
    return result

def transition_matrix(B, C):
    B = np.column_stack(B)
    C = np.column_stack(C)
    return np.linalg.inv(C) @ B


n = int(input("Enter dimension of vectors (n): "))
m = int(input("Enter number of vectors: "))

vectors = []
for i in range(m):
    v = np.array(list(map(float, input(f"Enter vector {i+1} (space-separated): ").split())))
    vectors.append(v)


independent, rank = check_linear_dependence(vectors)
print("\nLinear Dependence Check:")
print(f"Rank = {rank}")
print("Vectors are linearly INDEPENDENT." if independent else "Vectors are linearly DEPENDENT.")


coeffs = list(map(float, input("\nEnter coefficients for linear combination: ").split()))
comb = linear_combination(vectors, coeffs)
print("Linear Combination Result:", comb)


print("\nEnter two bases (each with n vectors) to find transition matrix:")
B = []
C = []
for i in range(n):
    B.append(np.array(list(map(float, input(f"Basis B vector {i+1}: ").split()))))
for i in range(n):
    C.append(np.array(list(map(float, input(f"Basis C vector {i+1}: ").split()))))

P = transition_matrix(B, C)
print("\nTransition Matrix (B → C):\n", P)

# 8.	Find the orthonormal basis of a given vector space using the Gram-Schmidt orthogonalization process.
import numpy as np

def gram_schmidt(vectors):
    """Return orthonormal basis using Gram-Schmidt process."""
    vectors = [np.array(v, dtype=float) for v in vectors]
    orthogonal = []
    orthonormal = []

    for v in vectors:
        for u in orthogonal:
            v = v - np.dot(v, u) / np.dot(u, u) * u
        
        if np.linalg.norm(v) < 1e-10:  # skip if dependent
            continue
        
        orthogonal.append(v)
        orthonormal.append(v / np.linalg.norm(v))
    
    return np.array(orthonormal)


n = int(input("Enter number of vectors: "))
m = int(input("Enter dimension of each vector: "))

vectors = []
print("\nEnter vectors (space-separated):")
for i in range(n):
    v = list(map(float, input(f"Vector {i+1}: ").split()))
    vectors.append(v)


orthonormal_basis = gram_schmidt(vectors)

print("\n--- Orthonormal Basis ---")
print(orthonormal_basis)


print("\n--- Verification: uᵀu (should be identity matrix) ---")
print(np.round(orthonormal_basis @ orthonormal_basis.T, 4))


