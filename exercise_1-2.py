# Cole McMullen
# Programming Project 1
# Excercise 2

import math
import numpy as np

# generate a matrix A for a given value of epsilon
def A_gen(epsilon):
	return np.array([[2, -2, 0],
			 [epsilon - 2, 2, 0],
			 [0, -1, 3]])


# generate the LU decomposition of a 3x3 matrix
def LU(arr):
	for i in range (0, 2):
		U[0, i] = arr[0, i]
		L[i, 0] = arr[i, 0]/U[0, 0]
	for i in range (1, 2):
		U[1, i] = (arr[1, i] - (L[1, 0] * U[0, i]))/L[1, 1]
	L[2, 1] = (arr[2, 1] - (L[2, 0]*U[0, 1]))/U[1, 1]
	U[2, 2] = (arr[2, 2] - (L[2, 0]*U[0, 2] + L[2, 1]*U[1, 2]))/L[2, 2]

# set the expected vector x to [1, 1, 1]
xex = np.ones(3)
print("X_exact = ")
print(xex)
# solve for x given b = [0, epsilon, 2] and find the error
# using epsilon = 10^-k, k = 0, 1, ..., 9
for k in range (0, 10):
	U = np.zeros((3, 3))
	L = np.eye(3)
	ep = pow(10, -1 * k)
	b = np.array([0, ep, 2]).T
	print("\nTesting with epsilon = "+str(ep)+"\n")
	A = A_gen(ep)
	LU(A)

	y = np.dot(np.linalg.inv(L), b)
	x = np.dot(np.linalg.inv(U), y)
	print("Error: ")
	print(x - xex)

# solve for x given b = [2log(5/2) - 2, (epsilon - 2)log(5/2) + 2, 2]
# and find the error
# using epsilon = 1/3 * 10^-k, k = 0, 1, ..., 9
xex = np.array([math.log(5/2), 1, 1]).T
print("\n\nX_exact = ")
print(xex)
for k in range (0, 10):
	U = np.zeros((3, 3))
	L = np.eye(3)
	ep = 1/3 * math.pow(10, -1*k)
	b = np.array([2 * math.log(2.5) - 2, (ep - 2) * math.log(2.5) + 2, 2]).T
	print("\nTesting epsilon = "+str(ep)+"\n")
	A = A_gen(ep)
	LU(A)

	y = np.dot(np.linalg.inv(L), b)
	x = np.dot(np.linalg.inv(U), y)
	print("Relative error: ")
	print(np.linalg.norm(x - xex)/np.linalg.norm(xex))

