# Cole McMullen
# Programming Project 2

import math
import time
import numpy as np

# general iterative method, after textbook
def itermeth(A, b, x0, tol, P):
	tic = time.perf_counter()
	dim = A.shape[0]
	if P == 'J': # if jacobi method is used
		L = np.diagflat(np.diag(A))
		U = np.eye(dim)
	if P == 'G': # if Gauss-Seidel method is used
		L = np.tril(A)
		U = np.eye(dim)
	beta = 1
	alpha = 1
	iter = 0;
	x = x0;
	r = b - np.dot(A, x0)
	r0 = np.linalg.norm(r)
	err = r0
	while ((err > tol)):
		z = np.linalg.solve(L, r)
		z = np.linalg.solve(U, z)
		iter = iter + 1
		x = x + (alpha * z)
		r = b - np.dot(A, x)
		err = np.linalg.norm(r) / r0
	toc = time.perf_counter()
	print("Elapsed time: " + str(toc - tic))
	print(iter)
	return x

# generate incomplete cholesky preconditioner for matrix A
def incompleteLU(A):
	dim = A.shape[0] - 1

	for k in range (1, dim):
		if(A[k, k] < 0):
			print("Negative Value: " + str(A[k, k]))
		A[k, k] = math.sqrt(A[k, k])
		for i in range (k+1, dim):
			if(A[i, k] != 0):
				A[i, k] = A[i, k]/A[k, k]
		for j in range (k+1, dim):
			for i in (j, dim):
				if(A[i, j] != 0):
					A[i, j] = A[i, j] - (A[i, k] * A[j, k])

	for i in (1, dim):
		for j in (i+1, dim):
			if(i < 9):
				A[i, j] = 0

	return A

# perform the Preconditioned Conjugate Gradient Method
def PCG(x, A, b, tol):
	iter = 0
	tic = time.perf_counter()
	P = incompleteLU(A) # generate preconditioner for A

	r = b - np.dot(A, x)
	E = np.linalg.norm(r)/np.linalg.norm(b) # stop if initial residual is small enough
	if(E < tol):
		return x
	z = np.linalg.solve(P, r)
	p = z
	result = PCG_helper(iter, x, A, b, P, r, r, z, p, tol)
	toc = time.perf_counter()
	print("Elapsed time: " + str(toc - tic))
	return result

# recursive helper function
def PCG_helper(iter, x, A, b, P, r, r0, z, p, tol):
	iter = iter + 1
	alpha = (np.dot(p.T, r))/(np.dot(p.T, np.dot(A, p)))
	x0 = x
	x = x + (alpha * p)
	r = r - (alpha * np.dot(A, p))
	E = np.linalg.norm(x - x0) # stop if difference in steps is small enough
	if(E < tol):
		print(iter)
		return x
	z = np.linalg.solve(P, r)
	beta = np.dot((np.dot(A, p).T), z)/np.dot(np.dot(A, p).T, p)
	p = z - (beta * p)

	return PCG_helper(iter, x, A, b, P, r, r0, z, p, tol)

# generate sample input
print("Generating Sample Input.....")
print("A:")
A = 2.001 * np.eye(30) + np.diag(np.ones(29), -1) + np.diag(np.ones(29), 1)
print(A)
print("b:")
x = np.ones((30, 1))
b = np.linalg.solve(A, x)
print(b)
x0 = np.zeros((30, 1))
# test jacobi method
print("Using the Jacobi method......")
print("Iterations:")
iterJ = itermeth(A, b, x0, pow(10, -12), 'J')
print("Error:")
print(np.linalg.norm(iterJ - x))
# test gauss-siedel
print("Using the Gauss-Siedel method.....")
print("Iterations:")
iterG = itermeth(A, b, x0, pow(10, -12), 'G')
print("Error:")
print(np.linalg.norm(iterG - x))
# test PCG with incomplete cholesky preconditioning
print("Using the PCG with an incomplete Cholesky preconditioner...")
print("Iterations:")
iterPCG = PCG(x0, A, b, pow(10, -12))
print("Error:")
print(np.linalg.norm(iterPCG - x))
