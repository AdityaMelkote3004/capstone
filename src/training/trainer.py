"""Generic trainer for LSTM and MLP baseline models."""

import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                              matthews_corrcoef, roc_auc_score, confusion_matrix)
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> Dict:
    m = {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'f1':       round(float(f1_score(y_true, y_pred,
                                         average='binary', zero_division=0)), 4),
        'mcc':      round(float(matthews_corrcoef(y_true, y_pred)), 4),
        'n_samples': int(len(y_true)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try:
            m['auc'] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            m['auc'] = 0.5
    return m


class Trainer:
    """Train LSTM or MLP models with early stopping on a dedicated val split."""

    def __init__(self, model, train_dataset, val_dataset, test_dataset,
                 lr=0.001, weight_decay=1e-4,
                 device='auto', save_dir='results'):
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                       if device == 'auto' else torch.device(device))
        self.model    = model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader   = DataLoader(val_dataset,   batch_size=256)
        self.test_loader  = DataLoader(test_dataset,  batch_size=256)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

    def _epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, preds, probs, targets = 0.0, [], [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                window = batch['window'].to(self.device)
                flat   = batch['flat'].to(self.device)
                target = batch['target'].to(self.device)

                logits = self.model(window=window, flat=flat)
                loss   = self.criterion(logits, target)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(target)
                p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                preds.extend(logits.argmax(1).cpu().numpy())
                probs.extend(p)
                targets.extend(target.cpu().numpy())

        m = compute_metrics(np.array(targets), np.array(preds), np.array(probs))
        m['loss'] = round(total_loss / len(loader.dataset), 6)
        return m

    def train(self, num_epochs=50, patience=10):
        best_mcc, wait = -1.0, 0
        for epoch in range(1, num_epochs + 1):
            tr = self._epoch(self.train_loader, train=True)
            vl = self._epoch(self.val_loader,   train=False)
            if epoch % 5 == 0 or epoch == 1:
                print(f"    Ep {epoch:3d} | Train Acc={tr['accuracy']:.3f} "
                      f"| Val Acc={vl['accuracy']:.3f} MCC={vl['mcc']:.4f}")
            if vl['mcc'] > best_mcc:
                best_mcc, wait = vl['mcc'], 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_dir, 'best_model.pt'))
            else:
                wait += 1
                if wait >= patience:
                    print(f"    Early stop at epoch {epoch}")
                    break

        self.model.load_state_dict(torch.load(
            os.path.join(self.save_dir, 'best_model.pt'), weights_only=True))
        test_m = self._epoch(self.test_loader, train=False)
        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump({'test': test_m, 'best_val_mcc': round(best_mcc, 4)}, f, indent=2)
        return test_m
