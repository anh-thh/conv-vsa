import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


def make_encoder(dim: int = 10_000) -> Callable:
    phases = np.random.default_rng(42).uniform(-np.pi, np.pi, dim)
    return lambda r: np.exp(1j * phases * r)


def function_representation(f: np.ndarray, encode: Callable) -> np.ndarray:
    """Compute vector representation for function f[r]"""
    return sum(f[r] * encode(r) for r in range(len(f)))


def retrieve(y_f: np.ndarray, s: int, encode: Callable) -> float:
    """Retrieve (approximate) f[s] from function representation y_f"""
    return float(np.real(y_f @ encode(s).conj()) / len(y_f))


def bind(y_f: np.ndarray, y_g: np.ndarray) -> np.ndarray:
    return y_f * y_g


def vsa_convolution(f: np.ndarray, g: np.ndarray, encode: Callable) -> np.ndarray:
    """Approximate convolution using VSA """
    y_bound = bind(function_representation(f, encode),
                   function_representation(g, encode))
    out_len = len(f) + len(g) - 1
    return np.array([retrieve(y_bound, u, encode) for u in range(out_len)])



if __name__ == "__main__":
    f = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    g = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    num_trials = 5

    errors = []
    dims = [i for i in range(100, 10000, 100)]

    for dim in dims:
        mses = []
        for _ in range(num_trials):
            encode = make_encoder(dim=dim)
            vsa_conv  = vsa_convolution(f, g, encode)
            true_conv = np.convolve(f, g, mode='full')
            mses.append(np.mean((true_conv - vsa_conv) ** 2))
        
        errors.append(np.mean(mses))

    plt.figure()
    plt.plot(dims, errors)
    plt.xlabel("Dimensions")
    plt.ylabel("MSE")
    plt.title("VSA Convolution Error vs Dimensions")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
