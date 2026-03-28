"""
Spatio-Temporal Momentum neural network model.
MLP that takes concatenated TSM + CSM features for all assets
and outputs position weights in [-1, 1] via tanh.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class SpatioTemporalMomentumNet(nn.Module):
    """MLP for joint TSM+CSM momentum signal generation.

    Args:
        n_features: Total number of input features (n_assets * n_feature_types).
        n_assets: Number of assets to produce signals for.
        hidden_size: Hidden layer size.
        n_hidden: Number of hidden layers.
        dropout: Dropout rate.
    """

    def __init__(self, n_features: int, n_assets: int, hidden_size: int = 64,
                 n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_size = n_features
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = hidden_size
        layers.append(nn.Linear(in_size, n_assets))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(features: np.ndarray, returns: np.ndarray, n_assets: int,
                epochs: int = 100, lr: float = 1e-3, hidden_size: int = 64,
                batch_size: int = 32) -> SpatioTemporalMomentumNet:
    """Train the Spatio-Temporal Momentum model.

    Uses a differentiable Sharpe-like loss (negative mean return / std of returns).

    Args:
        features: Training features array of shape (T, n_features).
        returns: Training forward returns array of shape (T, n_assets).
        n_assets: Number of assets.
        epochs: Number of training epochs.
        lr: Learning rate.
        hidden_size: Hidden layer size.
        batch_size: Mini-batch size.

    Returns:
        Trained model.
    """
    device = torch.device("cpu")
    n_features = features.shape[1]

    model = SpatioTemporalMomentumNet(
        n_features=n_features,
        n_assets=n_assets,
        hidden_size=hidden_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X = torch.tensor(features, dtype=torch.float32, device=device)
    Y = torch.tensor(returns, dtype=torch.float32, device=device)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            positions = model(xb)
            # Portfolio return for each time step: sum of (position * asset return)
            portfolio_ret = (positions * yb).sum(dim=1) / n_assets
            # Negative Sharpe-like loss
            if portfolio_ret.std() > 1e-8:
                loss = -(portfolio_ret.mean() / portfolio_ret.std())
            else:
                loss = -portfolio_ret.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def predict_positions(model: SpatioTemporalMomentumNet, features: np.ndarray) -> np.ndarray:
    """Generate position signals from trained model.

    Args:
        features: Feature array of shape (T, n_features).

    Returns:
        Position array of shape (T, n_assets) with values in [-1, 1].
    """
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        positions = model(X).numpy()
    return positions
