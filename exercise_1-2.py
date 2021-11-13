import math
import numpy as np

def A_gen(epsilon):
	return np.array([[2, -2, 0],
			 [epsilon - 2, 2, 0],
			 [0, -1, 3]])


def LU(arr):
	for i in range (0, 2):
		U[0, i] = arr[0, i]
		L[i, 0] = arr[i, 0]/U[0, 0]
	for i in range (1, 2):
		U[1, i] = (arr[1, i] - (L[1, 0] * U[0, i]))/L[1, 1]
	L[2, 1] = (arr[2, 1] - (L[2, 0]*U[0, 1]))/U[1, 1]
	U[2, 2] = (arr[2, 2] - (L[2, 0]*U[0, 2] + L[2, 1]*U[1, 2]))/L[2, 2]

xex = np.ones(3)
print("X_exact = ")
print(xex)
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

