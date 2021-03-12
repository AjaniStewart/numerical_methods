import numpy as np
from typing import Tuple



def create_tridiagonal(m: int, band: Tuple) -> np.ndarray:
  td = np.zeros((m,m))
  for i in range(len(td)):
    if i == 0:
      td[i,i:i+2] = band[1:]
    elif i == len(td)-1:
      td[i,i-1:i+1] = band[:2]
    else:
      td[i,i-1:i+2] = band
  return td

def make_b(m: int) -> np.ndarray:
  return np.array([1/i for i in range(1,m+1)])


# Exercise 3.8.3
def LU_factorization(A: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
  m = A.shape[1]
  U = A.copy()
  L = np.eye(m)
  for i in range(m-1):
    for j in range(i+1,m):
      L[j,i] = U[j,i] / U[i,i]
      U[j,i:m] = U[j,i:m] - L[j,i] * U[i,i:m]
  return L,U


# Exercise 3.8.4
def LUsolve(b: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
  m = len(b)
  # print(m)
  x = np.zeros(m)
  # print(x)
  # forward sub
  # solve Ux = L^-1b
  # initialize
  x[0] = b[0]/L[0,0]
  for i in range(1,m):
    x[i] = (b[i] - np.dot(L[i,:i],x[:i]))/L[i,i]
  #backward sub
  # solve x = U^{-1}L^{-1}b
  # init
  x[m-1] = x[m-1]/U[m-1,m-1]
  for i in range(m-2,-1,-1):
    x[i] = (x[i] - np.dot(U[i,i+1:],x[i+1:]))/U[i,i]

  return x

  # return x

# just to make sure the above is correct
def inversesolve(b: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
  # solve Ux = L^-1b
  y = np.linalg.inv(L) @ b
  # solve x = U^{-1}L^{-1}b
  x = np.linalg.inv(U) @ y
  return x

print("Exercise 3.8.4")
m = 10
A = create_tridiagonal(m,(0.25,0.5,0.25))
L,U = LU_factorization(A)
print('A')
print(A)
print()
print('L')
print(L)
print('U')
print(U)

b = make_b(m)

x_hat = LUsolve(b,L,U)
print(f"x_hat : {x_hat}")

#exercise 3.8.5
print()
print("Exercise 3.8.5")
m = 50
A = create_tridiagonal(m,(0.25,0.5,0.25))
L,U = LU_factorization(A)


b = make_b(m)

x_hat = LUsolve(b,L,U)
residual = np.linalg.norm((np.matmul(A,x_hat)-b))/np.linalg.norm(b)
print(f"residual : {residual}")
print()
print("Exercise 3.8.6")
evs = np.linalg.eigvals(A)
print(f"Jacobi and Gauss-Seidel will converge because A is convergent")
print(f"since σ(A) = {np.max(evs)} < 1")
print("and by construction of A, none of the elements on the diagonal are 0")

print()
print("Exercise 3.8.7: Jacobi")
# I have no idea why this takes thousands of iterations to converge
# but it runs fairly quickly either way
# and i think its pretty slick

def jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, 
            threshold: float, itermax: int) -> Tuple[np.ndarray, int]:
  n = 0
  x = x0
  while np.linalg.norm((A @ x - b))/np.linalg.norm(b) > threshold and n < itermax:
    # diagonals of A
    diag = np.diag(A)
    # skip the diag entries when computing the sum
    # then divide by those entries
    x_new = (b -(A-np.diag(diag)) @ x) / diag
    n += 1
    x = x_new

  return x_new, n

def jacobi2(A: np.ndarray, b: np.ndarray, x0: np.ndarray, 
            threshold: float, itermax: int) -> Tuple[np.ndarray, int]:
  # CONVERGES AT THE SAME RATE
  n = 0
  x = x0
  while np.linalg.norm((A @ x - b))/np.linalg.norm(b) > threshold and n < itermax:
    x_new = np.zeros_like(x)
    for i in range(A.shape[0]):
      x_new[i] = (b[i] - np.dot(A[i,:i],x[:i]) - np.dot(A[i,i+1:],x[i+1:])) / A[i,i]
    n += 1
    x = x_new

  return x, n


x0 = np.ones(m)
threshold = 1e-2
itermax = 10000

x_hat, itercount = jacobi(A,b,x0,threshold,itermax)
x_hat2, itercount2 = jacobi2(A,b,x0,threshold,itermax)

rmr1 = np.linalg.norm(A @ x_hat - b)/np.linalg.norm(b)
rmr2 = np.linalg.norm(A @ x_hat2 - b)/np.linalg.norm(b)
print(f"jacobi - relative normed residual: {rmr1} iterations: {itercount}")
print(f"jacobi2 - relative normed residual: {rmr2} iterations: {itercount2}")

print()
print("Exercise 3.8.8: Guass-Seidel")
def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: np.ndarray, 
            threshold: float, itermax: int) -> Tuple[np.ndarray, int]:
  n = 0
  m = A.shape[0]
  res = np.linalg.norm(A @ x0 - b)/np.linalg.norm(b) 
  while res > threshold and n < itermax:
    for i in range(m):
      s1 = np.dot(A[i,:i],x0[:i])
      s2 = np.dot(A[i,i+1:],x0[i+1:])
      x0[i] = (b[i]-s1-s2)/A[i,i]
    n += 1
    res = np.linalg.norm(A @ x0 - b)/np.linalg.norm(b) 
  return x0, n


x_hat, itercount = gauss_seidel(A,b,x0,threshold,itermax)
rmr = np.linalg.norm(A @ x_hat - b)/np.linalg.norm(b)
print(f"relative normed residual: {rmr} iterations: {itercount}")