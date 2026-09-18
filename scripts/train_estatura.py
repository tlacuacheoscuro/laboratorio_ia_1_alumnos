#!/usr/bin/env python3
"""Entrenamiento simple en PyTorch para predecir estatura a partir de la edad.

Uso:
  pip install -r requirements.txt  # contiene torch, pandas, numpy
  python3 scripts/train_estatura.py --data data/estatura_ninos.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_data(path):
    df = pd.read_csv(path)
    x = df[["age"]].values.astype(np.float32)
    y = df[["height"]].values.astype(np.float32)
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/estatura_ninos.csv")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="models/estatura_model.pth")
    args = p.parse_args()

    x, y = load_data(args.data)
    # normalizar entrada
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-8
    x_norm = (x - x_mean) / x_std

    X = torch.from_numpy(x_norm)
    Y = torch.from_numpy(y)

    dataset = TensorDataset(X, Y)
    n_test = int(0.2 * len(dataset))
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / n_train

        if epoch % 50 == 0 or epoch == 1:
            # evaluar
            model.eval()
            with torch.no_grad():
                test_losses = []
                maes = []
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    test_losses.append(criterion(pred, yb).item() * xb.size(0))
                    maes.append(torch.abs(pred - yb).mean().item() * xb.size(0))
                test_loss = sum(test_losses) / n_test
                mae = sum(maes) / n_test
            print(f"Epoch {epoch:03d}  Train MSE: {train_loss:.4f}  Test MSE: {test_loss:.4f}  Test MAE: {mae:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
    }, args.out)
    print(f"Modelo guardado en {args.out}")


if __name__ == "__main__":
    main()
