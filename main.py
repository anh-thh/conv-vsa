import torch
from typing import Callable
import matplotlib.pyplot as plt


def make_encoder(dim: int = 10_000, max_len: int = 1000):
    phases = torch.rand(dim) * (2 * torch.pi) - torch.pi    # [D, ]

    r = torch.arange(max_len).float().unsqueeze(1)          # [L, 1]
    encodings = torch.exp(1j * r * phases)                  # [L, D]

    return lambda r: encodings[r]


def function_representation(f: torch.Tensor, encode: Callable) -> torch.Tensor:
    """Compute vector representation for function f[r]"""
    assert len(f.shape)==1, f"Expect 1-D data but receive {len(f.shape)}"
    r = torch.arange(len(f))
    E = encode(r)               # [L, D]
    return (f.unsqueeze(1) * E).sum(dim=0)   # [D]


def retrieve(y_f: torch.Tensor, indices: torch.Tensor, encode: Callable):
    """Retrieve (approximate) f[s] from function representation y_f"""
    E = encode(indices)             # [L, D]
    return torch.real(E @ torch.conj(y_f)) / y_f.shape[0]


def bind(y_f: torch.Tensor, y_g: torch.Tensor) -> torch.Tensor:
    return y_f * y_g


def vsa_convolution(f: torch.Tensor, g: torch.Tensor, encode: Callable) -> torch.Tensor:
    """Approximate convolution using VSA """
    y_bound = bind(function_representation(f, encode),
                   function_representation(g, encode))
    out_len = len(f) + len(g) - 1
    indices = torch.arange(out_len)
    return retrieve(y_bound, indices, encode)


if __name__ == "__main__":
    f = torch.tensor([1, 2, 3, 4, 5, 4, 3, 2, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    g = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                      1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

    num_trials = 5
    errors = []
    dims = [i for i in range(100, 10000, 100)]

    max_len = len(f) + len(g)

    for dim in dims:
        mses = []
        for _ in range(num_trials):
            encode = make_encoder(dim=dim, max_len=max_len)
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
