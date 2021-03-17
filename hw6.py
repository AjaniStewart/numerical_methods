import numpy as np
import numpy.linalg as LA
import time

def gradient(x, y, z, w):
    # Calculated using mathematica
    g = np.array(
        [
            -10 - 164 * w + 154 * x - 2 * y + 30 * z,
            4 - 14 * w - 2 * x + 36 * y - 2 * z,
            -30 - 116 * w + 30 * x - 2 * y + 118 * z,
            30 + 246 * w - 164 * x - 14 * y - 116 * z,
        ]
    )
    return g

# step 0
def g(x, y, z, w):
    # return ||L(x,y,z,w) - b|| ^ 2
    # done using mathematica
    return (
        4
        + 30 * w
        + 123 * w ** 2
        - 10 * x
        - 164 * w * x
        + 77 * x ** 2
        + 4 * y
        - 14 * w * y
        - 2 * x * y
        + 18 * y ** 2
        - 30 * z
        - 116 * w * z
        + 30 * x * z
        - 2 * y * z
        + 59 * z ** 2
    )

def max_grad_descent(x0: np.ndarray, thresh: float, max_iter: int) -> np.ndarray:
    # 0 if f : R^n -> R^m, minimize g : ||f - b|| ^ 2
    # so g : R^n -> R
    # 1 compute gradient of g (A)
    # 2 set h(t) = g(x_i - t*grad(g,x_i))
    # 3 minimize h
    # 4 x_{i+1} = x_i - t_0*grad(g,x_i)
    iter_count = 0
    x = x0
    while LA.norm(A @ x - b) / LA.norm(b) > thresh and iter_count < max_iter:
        # step 1
        grad = gradient(*x)
        # start of black magic (steps 2 and 3)
        polys = [np.poly1d((-grad[i], x[i])) for i in range(len(grad))]
        # h(t) = g(x-t*grad)
        h = g(*polys)
        # find roots of derivative of h == find min
        t0 = h.deriv().r
        # end of black magic
        # step 4
        x = x - t0[0] * grad
        iter_count += 1
    return x, iter_count


print("Exercise 4.3.3")

A = np.array([[4, -2, 3, -5], [3, 3, 5, -8], [-6, -1, 4, 3], [-4, 2, -3, 5]])

x0 = np.array([5, 5, 5, 5])
b = np.array([1, 1, 1, -1])
thresh = 1e-4
max_iter = 100

print(f"using x0 = {x0}")
tic = time.perf_counter()
x_hat, iter_count = max_grad_descent(x0, thresh, max_iter)
toc = time.perf_counter()
print(f"max_grad_descent done in {toc - tic:0.4f} seconds")
print(f"x0: {x0}")
rmr = LA.norm(A @ x_hat - b) / LA.norm(b)
print(f"relative normed residual: {rmr} iterations: {iter_count}")
print(f"x_hat: {x_hat}")
print(A @ x_hat)

print("\n")

x0 = np.array([1, 2, 3, 4])
print(f"using x0 = {x0}")
tic = time.perf_counter()
x_hat, iter_count = max_grad_descent(x0, thresh, max_iter)
toc = time.perf_counter()
print(f"max_grad_descent done in {toc - tic:0.4f} seconds")
print(f"x0: {x0}")
rmr = LA.norm(A @ x_hat - b) / LA.norm(b)
print(f"relative normed residual: {rmr} iterations: {iter_count}")
print(f"x_hat: {x_hat}")
print(A @ x_hat)
