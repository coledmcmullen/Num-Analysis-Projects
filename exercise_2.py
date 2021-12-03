# Cole McMullen
# Programming Project 2

import math
import numpy as np

# general iterative method, after textbook
def itermeth(A, b, x0, tol, P):
	dim = A.shape[0]
	if P == 'J':
		L = np.diagflat(np.diag(A))
		U = np.eye(dim)
	if P == 'G':
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
	return iter

# generate preconditioner for A
def precondition(A):
	return np.diag(np.diag(A))
	#return np.tril(A)

# perform the Preconditioned Conjugate Gradient Method
def PCG(x, A, b, tol):
	iter = 0
	P = precondition(A)

	r = b - np.dot(A, x)
	E = np.linalg.norm(r)/np.linalg.norm(b) # stop if initial residual is small enough
	if(E < tol):
		return x
	z = np.linalg.solve(P, r)
	p = z
	result = PCG_helper(iter, x, A, b, P, r, r, z, p, tol)
	return result

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

# test Jacobi method on given matrix
A = 3 * np.eye(10) - 2 * np.diag(np.ones(9), -1) - np.diag(np.ones(9), 1)
x = np.ones((10, 1))
b = np.linalg.solve(A, x)
x0 = np.zeros((10, 1))
iterJ = itermeth(A, b, x0, pow(10, -12), 'J')
#iterJ = jacobi(A, b, x0, pow(10, -12))
iterG = itermeth(A, b, x0, pow(10, -12), 'G')
iterPCG = PCG(x0, A, b, pow(10, -12))
#print(A)
#print(x)
#print(b)
print(iterJ)
print(iterG)
