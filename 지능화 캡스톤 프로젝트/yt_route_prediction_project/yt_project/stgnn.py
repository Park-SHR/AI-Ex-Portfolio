from __future__ import annotations
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .evaluate import regression_metrics, congestion_metrics


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # x: [B,N,F], a_norm: [N,N]
        x = torch.einsum("ij,bjf->bif", a_norm, x)
        return self.linear(x)


class SimpleSTGNN(nn.Module):
    """Lightweight ST-GNN for pre-Meta K-Port validation.

    It is not a full ASTGCN reproduction. It verifies the project pipeline:
    link adjacency A + time-window feature X -> future link count/congestion Y.
    """
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.gcn = GraphConv(n_features, hidden)
        self.temporal = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,F], a_norm: [N,N]
        # Vectorized graph aggregation over all time steps.
        # [B,T,F,N] @ [N,N] -> [B,T,F,N] -> [B,T,N,F]
        x_agg = torch.matmul(x.permute(0, 1, 3, 2), a_norm.T).permute(0, 1, 3, 2)
        z = self.act(self.gcn.linear(x_agg))  # [B,T,N,H]
        # Fast temporal aggregation for validation: recent window mean + MLP.
        z = z.mean(dim=1)  # [B,N,H]
        z = self.temporal(z)
        y = self.head(z).squeeze(-1)  # [B,N]
        return y.clamp(min=0)


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    deg = A.sum(axis=1)
    deg[deg < 1e-6] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return d_inv_sqrt @ A @ d_inv_sqrt


def train_stgnn_model(tensor_ds: dict, A_df: pd.DataFrame, output_model_dir, epochs: int = 12, hidden: int = 32, threshold: float = 0.55) -> tuple[pd.DataFrame, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    link_ids = tensor_ds["link_ids"]
    A = A_df.loc[link_ids, link_ids].values.astype(np.float32)
    A_norm = torch.tensor(normalize_adjacency(A), dtype=torch.float32, device=device)
    X_train = torch.tensor(tensor_ds["X_train"], dtype=torch.float32)
    y_train = torch.tensor(tensor_ds["y_train"], dtype=torch.float32)
    X_valid = torch.tensor(tensor_ds["X_valid"], dtype=torch.float32)
    y_valid = torch.tensor(tensor_ds["y_valid"], dtype=torch.float32)
    X_test = torch.tensor(tensor_ds["X_test"], dtype=torch.float32)
    y_test = torch.tensor(tensor_ds["y_test"], dtype=torch.float32)

    n_features = X_train.shape[-1]
    model = SimpleSTGNN(n_features=n_features, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    best_valid = float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, A_norm)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            vpred = model(X_valid.to(device), A_norm)
            vloss = loss_fn(vpred, y_valid.to(device)).item()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": float(vloss)})
        if vloss < best_valid:
            best_valid = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device), A_norm).detach().cpu().numpy()
    torch.save({"state_dict": model.state_dict(), "link_ids": link_ids, "history": history}, output_model_dir / "SimpleSTGNN.pt")
    reg = regression_metrics(tensor_ds["y_test"], pred)
    # Estimate congestion score from normalized predicted future count.
    denom = max(np.percentile(tensor_ds["y_train"], 90), 1.0)
    true_score = np.clip(tensor_ds["y_test"] / denom, 0, 1)
    pred_score = np.clip(pred / denom, 0, 1)
    cong = congestion_metrics(true_score, pred_score, threshold=threshold)
    metrics = pd.DataFrame([{"model": "SimpleSTGNN", **reg, **cong, "device": str(device), "epochs": epochs}])
    return metrics, pred
