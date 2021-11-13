import math

def function(x):
	a = 230
	b = 18
	c = 9
	d = -221
	e = -9

	return (a*math.pow(x, 4)) + (b*math.pow(x, 3)) + (c*math.pow(x, 2)) + (d*x) + e

def function_prime(x):
	a = 230
	b = 18
	c = 9
	d = -221

	return (a*4*math.pow(x, 3)) + (b*3*math.pow(x, 2)) + (c*2*x) + d

def secant_method(func, x_k, x_ksub1, err):
	f_x_k = func(x_k)
	f_x_ksub1 = func(x_ksub1)
	if(abs(f_x_k) < err):
		return x_k
	if(abs(f_x_ksub1) < err):
		return x_ksub1
	x_kplus1 = x_k - f_x_k/((f_x_k - f_x_ksub1)/(x_k - x_ksub1))
	return secant_method(func, x_kplus1, x_k, err)

def newton_method(func, func_prime, x, err):
	f_x = func(x)
	f_prime_x = func_prime(x)
	if(abs(f_x) < err):
		return x
	new_x = x - (f_x/f_prime_x)
	return newton_method(func, func_prime, new_x, err)

print("Finding result in [-1, 0] using secant method...")
result = secant_method(function, -1, 0, math.pow(10, -6))
print(result)

# is there a way to make this work by changing my code?
print("Finding result in [0, 1] using secant method...")
result = secant_method(function, 0.8, 1.2, math.pow(10, -6))
print(result)

print("Finding result in [-1, 0] using newton method...")
result = newton_method(function, function_prime, -1, math.pow(10, -6))
print(result)

print("Finding result in [0, 1] using newton method...")
result = newton_method(function, function_prime, 1, math.pow(10, -6))
print(result)
