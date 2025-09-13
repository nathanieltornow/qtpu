# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import quimb as qu
import cotengra as ctg
import quimb.tensor as qtn


# -------------------- target function (same as before) --------------------
def true_fn(x):  # x: [B, L]
    s = x.sum(dim=1)
    return s


# -------------------- MPS template (phys dim d=2) ------------------------
def build_mps_template(L: int, D: int, d: int = 2):
    mps = qtn.MPS_rand_state(L, bond_dim=D, phys_dim=d, cyclic=False)
    # rename physical indices to k{i}
    for i in range(L):
        old = mps.site_ind(i)
        mps = mps.reindex_({old: f"k{i}"})
    # rename virtual bonds to b{i}
    for i in range(L):
        T = mps[i]
        other = [ix for ix in T.inds if ix != f"k{i}"]
        if i == 0:
            mps = mps.reindex_({other[0]: "b0"})
        elif i == L - 1:
            mps = mps.reindex_({other[0]: f"b{L - 2}"})
        else:
            left_ix, right_ix = other
            mps = mps.reindex_({left_ix: f"b{i - 1}", right_ix: f"b{i}"})
    return mps


# -------------------- Learnable sin/cos feature map ----------------------
class SinCosFeatures(nn.Module):
    """
    Per-site features: [cos(ω_i x_i), sin(ω_i x_i)] -> d=2
    If you want fixed frequencies, set learn_freq=False.
    """

    def __init__(self, L: int, init_scale: float = 1.0, learn_freq: bool = True):
        super().__init__()
        self.L = L
        w = init_scale * torch.ones(L)  # start with ω_i = 1
        self.omega = nn.Parameter(w) if learn_freq else nn.Parameter(w, requires_grad=False)

    def forward(self, x):  # x: [B, L] or [L]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, L = x.shape
        assert L == self.L
        wx = x * self.omega  # [B, L]
        # features per site: stack along last dim -> [B, L, 2]
        phi = torch.stack([torch.cos(wx), torch.sin(wx)], dim=-1)
        return phi


# -------------------- Packed MPS model (quimb pack/unpack) ---------------
class MPSRegressor(nn.Module):
    """
    Holds packed MPS parameters and evaluates on a batch.
    Expects features phi: [B, L, 2] with d=2.
    """

    def __init__(self, mps_template: qtn.TensorNetwork, L: int):
        super().__init__()
        self.L = L
        params, self.skel = qtn.pack(mps_template)
        # safe parameter init from existing torch/numpy arrays
        self.params = nn.ParameterDict({
            str(k): nn.Parameter(torch.as_tensor(v, dtype=torch.float32).detach().clone()) for k, v in params.items()
        })
        self.out_bias = nn.Parameter(torch.zeros(()))  # small bias helps
        self.pathopt = ctg.ReusableHyperOptimizer(
            max_repeats=16,  # search effort
            progbar=False,
            parallel=False,
            minimize="flops",
        )

    def _predict_one(self, phi_row: torch.Tensor) -> torch.Tensor:
        """
        phi_row: [L, 2] features for one sample
        """
        arrays = {int(k): p for k, p in self.params.items()}
        tn = qtn.unpack(arrays, self.skel)

        # attach feature vectors on each physical leg k{i}
        feats = [qtn.Tensor(data=phi_row[i], inds=[f"k{i}"]) for i in range(self.L)]
        tn_all = tn | qtn.TensorNetwork(feats)

        out = tn_all.contract(all, backend="torch", optimize=self.pathopt)
        return out

    def forward(self, phi_batch: torch.Tensor) -> torch.Tensor:
        if phi_batch.dim() == 2:
            phi_batch = phi_batch.unsqueeze(0)
        preds = [self._predict_one(phi_batch[b]) for b in range(phi_batch.shape[0])]
        return torch.stack(preds, dim=0).squeeze(-1) + self.out_bias


# -------------------- Training loop --------------------------------------
def main():
    torch.set_float32_matmul_precision("high")  # fine on CPU

    # data
    L = 12
    D = 32  # a bit more capacity than before
    d = 2  # phys dim = 2 (sin/cos)
    Ntr, Nval = 4096, 1024

    # sample inputs in [-π, π] and normalize to ~[-1, 1] for stability
    x_train = (2 * math.pi) * (torch.rand(Ntr, L) - 0.5)
    x_val = (2 * math.pi) * (torch.rand(Nval, L) - 0.5)
    x_train = x_train / math.pi
    x_val = x_val / math.pi

    y_train = true_fn(x_train * math.pi)  # IMPORTANT: undo the scale inside true_fn if you want same target
    y_val = true_fn(x_val * math.pi)  # so target still sees the original range

    # models
    features = SinCosFeatures(L=L, init_scale=1.0, learn_freq=True)
    mps_tn = build_mps_template(L=L, D=D, d=d)
    mps_tn.apply_to_arrays(lambda a: torch.tensor(a, dtype=torch.float32))
    model = MPSRegressor(mps_tn, L=L)

    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(list(features.parameters()) + list(model.parameters()), lr=1e-3, weight_decay=1e-4)

    B = 256
    steps = 400
    for step in range(steps):
        idx = torch.randint(0, Ntr, (B,))
        xb, yb = x_train[idx], y_train[idx]

        phi = features(xb)  # [B, L, 2]
        pred = model(phi)  # [B]
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        opt.step()

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                val = loss_fn(model(features(x_val)), y_val).item()
            print(f"step {step + 1:3d} | train {loss.item():.4f} | val {val:.4f}")

    with torch.no_grad():
        val = loss_fn(model(features(x_val)), y_val).item()
    print(f"Final val MSE: {val:.4f}")

    # do an example prediction
    x_ex = torch.linspace(-1, 1, 200).unsqueeze(1)  # [200, 1]
    x_ex = x_ex.repeat(1, L)  # [200, L]
    with torch.no_grad():
        y_ex = true_fn(x_ex * math.pi)
        y_pred = model(features(x_ex)).squeeze(-1)

    import matplotlib.pyplot as plt

    plt.plot(x_ex[:, 0].numpy(), y_ex.numpy(), label="true")
    plt.plot(x_ex[:, 0].numpy(), y_pred.numpy(), label="pred")
    plt.legend()
    plt.title("MPS regression with sin/cos features")
    plt.savefig("mps_regression.png", dpi=200)


if __name__ == "__main__":
    main()
