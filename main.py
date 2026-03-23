import torch
from typing import Callable
import matplotlib.pyplot as plt

class VSAConv1d:
    def __init__(self, dim: int = 10000, max_len: int = 1000):
        self.dim = dim
        self.max_len = max_len
        self.z_encodings = self._make_assoc_mem()

    def _make_assoc_mem(self):
        """
        Create associative memory to store z(r) for r in {1, 2, ..., max_len}
        """
        phases = torch.rand(self.dim) * (2 * torch.pi) - torch.pi    # [D, ]

        r = torch.arange(self.max_len).float().unsqueeze(1)           # [L, 1]
        encodings = torch.exp(1j * r * phases)                        # [L, D]

        return encodings

    def function_representation(self, f: torch.Tensor) -> torch.Tensor:
        """Compute vector representation for function f[r]"""
        assert f.dim() == 1, f"Expect 1-D data but receive {f.dim()}"

        Z = self.z_encodings[:len(f)]                  # [L, D]
        return (f.unsqueeze(1) * Z).sum(dim=0)         # [D]

    def retrieve(self, y_f: torch.Tensor, indices: torch.Tensor):
        """Retrieve (approximate) f[s] from function representation y_f"""
        Z = self.z_encodings[indices]                  # [L, D]
        return torch.real(Z @ torch.conj(y_f)) / self.dim

    def bind(self, y_f: torch.Tensor, y_g: torch.Tensor) -> torch.Tensor:
        return y_f * y_g

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Approximate convolution using VSA """
        y_bound = self.bind(
            self.function_representation(f),
            self.function_representation(g),
        )
        out_len = len(f) + len(g) - 1
        indices = torch.arange(out_len)
        return self.retrieve(y_bound, indices)


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
            vsa_conv_layer = VSAConv1d(dim=dim, max_len=max_len)
            vsa_conv  = vsa_conv_layer.forward(f, g)

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
