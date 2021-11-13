# Cole McMullen
# Programming Project 1
# Exercise 1

import math

# apply the given function to an input
def function(x):
	a = 230
	b = 18
	c = 9
	d = -221
	e = -9

	return (a*math.pow(x, 4)) + (b*math.pow(x, 3)) + (c*math.pow(x, 2)) + (d*x) + e

# apply the derivative of the function to an input
def function_prime(x):
	a = 230
	b = 18
	c = 9
	d = -221

	return (a*4*math.pow(x, 3)) + (b*3*math.pow(x, 2)) + (c*2*x) + d

# apply the secant method to a function given two starting points and a tolerance
def secant_method(func, x_k, x_ksub1, err):
	f_x_k = func(x_k)
	f_x_ksub1 = func(x_ksub1)
	if(abs(f_x_k) < err):
		return x_k
	if(abs(f_x_ksub1) < err):
		return x_ksub1
	x_kplus1 = x_k - f_x_k/((f_x_k - f_x_ksub1)/(x_k - x_ksub1))
	return secant_method(func, x_kplus1, x_k, err)

# apply newton's method to a function given its derivative, a starting point and a tolerance
def newton_method(func, func_prime, x, err):
	f_x = func(x)
	f_prime_x = func_prime(x)
	if(abs(f_x) < err):
		return x
	new_x = x - (f_x/f_prime_x)
	return newton_method(func, func_prime, new_x, err)

# apply the methods to the desired intervals
print("Finding result in [-1, 0] using secant method...")
result = secant_method(function, -1, 0, math.pow(10, -6))
print(result)

# just providing 0 and 1 as starting values finds the result in
# [-1, 0], so closer starting points have to be used
print("Finding result in [0, 1] using secant method...")
result = secant_method(function, 0.8, 1, math.pow(10, -6))
print(result)

print("Finding result in [-1, 0] using newton method...")
result = newton_method(function, function_prime, -1, math.pow(10, -6))
print(result)

print("Finding result in [0, 1] using newton method...")
result = newton_method(function, function_prime, 1, math.pow(10, -6))
print(result)
