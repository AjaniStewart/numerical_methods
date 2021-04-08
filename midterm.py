import numpy as np
import numpy.linalg as LA
from typing import Tuple

#### HELPER FUNCTIONS I MADE from hw 5

def make_tridiagonal(m: int, band: Tuple) -> np.ndarray:
    td = np.zeros((m, m))
    for i in range(len(td)):
        if i == 0:
            td[i, i : i + 2] = band[1:]
        elif i == len(td) - 1:
            td[i, i - 1 : i + 1] = band[:2]
        else:
            td[i, i - 1 : i + 2] = band
    return td


def make_b(m: int) -> np.ndarray:
    return np.array([1 / i for i in range(1, m + 1)])


### -----------


def max_grad_descent(
    A: np.ndarray, x0: np.ndarray, b: np.ndarray, thresh: float, max_iter: int
) -> np.ndarray:
    x = x0
    r = b - A @ x
    norm_r = LA.norm(r)
    iter_count = 0
    while norm_r / LA.norm(b) >= thresh and iter_count < max_iter:
        alpha = (norm_r ** 2) / np.dot(r, A @ r)
        x = x + alpha * r
        r = b - A @ x
        norm_r = LA.norm(r)
        iter_count += 1
    return x, iter_count


def conjugate_grad(
    A: np.ndarray, x0: np.ndarray, b: np.ndarray, thresh: float, max_iter: int
) -> np.ndarray:
    x = x0
    r = b - A @ x
    p = r
    norm_r = LA.norm(r)
    iter_count = 0
    while norm_r / LA.norm(b) >= thresh and iter_count < max_iter:
        q = A @ p
        alpha = (norm_r ** 2) / np.dot(p,q)
        x = x + alpha * p
        r = r - alpha * q
        prev_norm_r = norm_r 
        norm_r = LA.norm(r)
        beta = (norm_r / prev_norm_r) ** 2
        p = r + beta * p
        iter_count += 1
    return x, iter_count

    


def main():
    m = 50
    A = make_tridiagonal(m, (0.25, 0.5, 0.25))
    b = make_b(m)
    x0 = np.ones(m)
    thresh = 1e-2
    max_iter = 100
    _, iter_count_max = max_grad_descent(A, x0, b, thresh, max_iter)
    print(f"Exercise 5.4.2")
    print(f"maximal gradient descent took {iter_count_max} iterations to converge")

    _, iter_count_conj = conjugate_grad(A, x0, b, thresh, max_iter)
    print(f"Exercise 5.4.3")
    print(f"conjugate gradient method took {iter_count_conj} iterations to converge")

    thresh = 1e-12
    _, iter_count_conj = conjugate_grad(A, x0, b, thresh, max_iter)
    print(f"Exercise 5.4.4")
    print(f"conjugate gradient method took {iter_count_conj} iterations to converge with threshold = 1e-12")


if __name__ == "__main__":
    main()
