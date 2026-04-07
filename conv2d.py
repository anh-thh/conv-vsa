import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class VSAConv2d:
    def __init__(self, dim: int = 10000, max_h: int = 100, max_w: int = 100):
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.z_row, self.z_col = self._make_assoc_mem()

    def _make_assoc_mem(self):
        """
        Create associative memory using 2D FPE; z(r1, r2) = z(r1) * z(r2)
        """
        phases_row = torch.rand(self.dim) * (2 * torch.pi) - torch.pi  # [D]
        phases_col = torch.rand(self.dim) * (2 * torch.pi) - torch.pi  # [D]

        r1 = torch.arange(self.max_h).float().unsqueeze(1)  # [max_h, 1]
        r2 = torch.arange(self.max_w).float().unsqueeze(1)  # [max_w, 1]

        z_row = torch.exp(1j * r1 * phases_row)  # [max_h, D]
        z_col = torch.exp(1j * r2 * phases_col)  # [max_w, D]

        return z_row, z_col

    def function_representation(self, f: torch.Tensor) -> torch.Tensor:
        """Compute vector representation for 2D function f[r1, r2]"""
        assert f.dim() == 2, f"Expect 2-D data but receive {f.dim()}"
        H, W = f.shape

        z_r = self.z_row[:H]    # [H, D]
        z_c = self.z_col[:W]    # [W, D]

        Z = z_r.unsqueeze(1) * z_c.unsqueeze(0)     # [H, W, D]

        return (f.unsqueeze(-1) * Z).sum(dim=(0, 1))    # [D]

    def retrieve(self, y_f: torch.Tensor, h_indices: torch.Tensor, w_indices: torch.Tensor) -> torch.Tensor:
        """Retrieve (approximate) f[s1, s2] from function representation y_f"""
        z_r = self.z_row[h_indices]     # [out_H, D]
        z_c = self.z_col[w_indices]     # [out_W, D]

        Z = z_r.unsqueeze(1) * z_c.unsqueeze(0)     # [out_H, out_W, D]

        return torch.real(Z @ torch.conj(y_f)) / self.dim   # [out_H, out_W]

    def bind(self, y_f: torch.Tensor, y_g: torch.Tensor) -> torch.Tensor:
        return y_f * y_g

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Approximate 2D convolution using VSA"""
        y_bound = self.bind(
            self.function_representation(f),
            self.function_representation(g),
        )
        H_f, W_f = f.shape
        H_g, W_g = g.shape
        out_H = H_f + H_g - 1
        out_W = W_f + W_g - 1
        h_indices = torch.arange(out_H)
        w_indices = torch.arange(out_W)
        return self.retrieve(y_bound, h_indices, w_indices)


if __name__ == "__main__":
    f = torch.zeros(10, 10)
    f[2:7, 2:7] = 1.0

    g = torch.zeros(5, 5)
    g[1:4, 1:4] = 1.0

    num_trials = 5
    errors = []
    dims = [i for i in range(100, 10000, 500)]

    max_h = f.shape[0] + g.shape[0]
    max_w = f.shape[1] + g.shape[1]

    for dim in dims:
        mses = []
        for _ in range(num_trials):
            vsa_conv_layer = VSAConv2d(dim=dim, max_h=max_h, max_w=max_w)
            vsa_conv = vsa_conv_layer.forward(f, g)

            true_conv = F.conv2d(
                f.view(1, 1, *f.shape),
                g.flip([0, 1]).view(1, 1, *g.shape),
                padding=(g.shape[0] - 1, g.shape[1] - 1),
            ).squeeze()

            mses.append(torch.mean((true_conv - vsa_conv) ** 2).item())

        errors.append(sum(mses) / len(mses))

    plt.figure()
    plt.plot(dims, errors)
    plt.xlabel("Dimensions")
    plt.ylabel("MSE")
    plt.title("VSA 2D Convolution Error vs Dimensions")
    plt.yscale("log")
    plt.grid(True)
    plt.show()
