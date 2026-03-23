import torch
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


def make_encoder(dim: int = 10_000) -> Callable:
    phases = torch.rand(dim) * (2 * torch.pi) - torch.pi
    return lambda r: torch.exp(1j * phases * r)


def function_representation(f: torch.Tensor, encode: Callable) -> torch.Tensor:
    """Compute vector representation for function f[r]"""
    return sum(f[r] * encode(r) for r in range(len(f)))


def retrieve(y_f: torch.Tensor, s: int, encode: Callable) -> float:
    """Retrieve (approximate) f[s] from function representation y_f"""
    return float(torch.real(torch.dot(y_f, torch.conj(encode(s)))) / len(y_f))


def bind(y_f: torch.Tensor, y_g: torch.Tensor) -> torch.Tensor:
    return y_f * y_g


def vsa_convolution(f: torch.Tensor, g: torch.Tensor, encode: Callable) -> torch.Tensor:
    """Approximate convolution using VSA """
    y_bound = bind(function_representation(f, encode),
                   function_representation(g, encode))
    out_len = len(f) + len(g) - 1
    return torch.tensor([retrieve(y_bound, u, encode) for u in range(out_len)])


if __name__ == "__main__":
    f = torch.tensor([1, 2, 3, 4, 5, 4, 3, 2, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    g = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                      1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

    num_trials = 5

    errors = []
    dims = [i for i in range(100, 10000, 100)]

    for dim in dims:
        mses = []
        for _ in range(num_trials):
            encode = make_encoder(dim=dim)
            vsa_conv  = vsa_convolution(f, g, encode)

            true_conv = torch.nn.functional.conv1d(
                f.view(1, 1, -1),
                g.flip(0).view(1, 1, -1),
                padding=g.numel() - 1
            ).view(-1)

            mses.append(torch.mean((true_conv - vsa_conv) ** 2).item())
        
        errors.append(sum(mses) / len(mses))

    plt.figure()
    plt.plot(dims, errors)
    plt.xlabel("Dimensions")
    plt.ylabel("MSE")
    plt.title("VSA Convolution Error vs Dimensions")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
