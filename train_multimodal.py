import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    precision_recall_curve
)

from featurize.featurize_multi import featurize_multi_model
from model.models import FusionMLPClassifier
from load_data.splits import group_stratified_kfold_split
import joblib


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ====== 评估 / 辅助函数 ======
def evaluate(model, X_list, y, device, threshold=0.5, return_loss=False, criterion=None):
    model.eval()
    with torch.no_grad():
        inputs = [torch.tensor(X, dtype=torch.float32).to(device) for X in X_list]
        logits = model(*inputs)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y, probs)
    aupr = average_precision_score(y, probs)
    f1 = f1_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)

    val_loss = None
    if return_loss and criterion is not None:
        labels = torch.tensor(y, dtype=torch.float32).to(device)
        val_loss = criterion(logits, labels).item()

    return auc, aupr, f1, precision, recall, probs, preds, val_loss


def find_best_threshold(y_true, probs):
    ps, rs, ts = precision_recall_curve(y_true, probs)
    ts = np.clip(ts, 1e-6, 1 - 1e-6)
    # drop first element (precision for threshold=inf)
    ps, rs, ts = ps[1:], rs[1:], ts[1:]
    f1s = 2 * ps * rs / (ps + rs + 1e-12)
    idx = np.nanargmax(f1s)
    return float(ts[idx]), float(ps[idx]), float(rs[idx])


def noise_std_at_epoch(epoch: int, total_epochs: int, start: float, end: float, mode: str = "linear") -> float:
    if total_epochs <= 1:
        return end
    t = epoch / (total_epochs - 1)
    if mode == "linear":
        return (1.0 - t) * start + t * end
    elif mode == "exp":
        eps = 1e-8
        s = max(start, eps)
        e = max(end, eps)
        r = e / s
        return s * (r ** t)
    else:
        return (1.0 - t) * start + t * end


def predict_proba_batched(model, X_list, device, batch_size=4096):
    model.eval()
    n = X_list[0].shape[0]
    probs_all = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            inputs = [torch.tensor(X[start:end], dtype=torch.float32).to(device) for X in X_list]
            logits = model(*inputs)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            probs_all[start:end] = probs

    return probs_all


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # ------------------ 超参 ------------------
    data_csv = "data/dataset.csv"
    smiles_col = "smiles"

    pre_model = ["Mol2Vec", "ChemBERTa"]

    hidden_dim = 256
    dropout = 0.1
    lr = 1e-4
    weight_decay = 1e-2
    batch_size = 128
    max_epochs = 50
    patience = 10

    initial_noise_std = 0
    final_noise_std = 0
    noise_mode = "linear"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ------------------ 载入数据 & 特征 ------------------
    df = pd.read_csv(data_csv)
    embs, y = featurize_multi_model(df, smiles_col=smiles_col, models=pre_model)
    X_emb1 = embs[0]
    X_emb2 = embs[1]

    N = len(y)
    assert X_emb1.shape[0] == N and X_emb2.shape[0] == N

    # test_df = pd.read_csv("data/test_dataset.csv")
    # embs_test_ind, y_test_ind = featurize_multi_model(test_df, models=pre_model)

    # ------------------ CV split ------------------
    kf = group_stratified_kfold_split(df)

    all_metrics = {"AUC": [], "AUPR": [], "F1": [], "Precision": [], "Recall": []}

    # StandardScaler
    scaler_emb1 = StandardScaler()
    scaler_emb2 = StandardScaler()

    all_fold_probs = []
    for fold, (train_idx, test_idx) in enumerate(kf, 1):
        print(f"\n=== Fold {fold} ===")

        X_emb1_tr, X_emb1_test = X_emb1[train_idx], X_emb1[test_idx]
        X_emb2_tr, X_emb2_test = X_emb2[train_idx], X_emb2[test_idx]
        y_tr, y_test = y[train_idx], y[test_idx]

        X_emb1_tr = scaler_emb1.fit_transform(X_emb1_tr)
        X_emb1_test = scaler_emb1.transform(X_emb1_test)

        X_emb2_tr = scaler_emb2.fit_transform(X_emb2_tr)
        X_emb2_test = scaler_emb2.transform(X_emb2_test)

        # DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_emb1_tr, dtype=torch.float32),
            torch.tensor(X_emb2_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)
        )

        # 计算每个样本的权重
        class_sample_count = np.bincount(y_tr.astype(int))  # [num_neg, num_pos]
        num_samples = len(y_tr)
        weights = np.zeros(num_samples, dtype=np.float32)
        weights[y_tr == 0] = 0.5
        weights[y_tr == 1] = class_sample_count[0] / class_sample_count[1]  # 负/正比例

        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

        # 构建模型
        embedding1_dim = X_emb1_tr.shape[1]
        embedding2_dim = X_emb2_tr.shape[1]
        model = FusionMLPClassifier(
            embedding1_dim=embedding1_dim,
            embedding2_dim=embedding2_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)

        criterion = FocalLoss(alpha=0.5, gamma=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 训练循环
        train_losses, val_losses = [], []
        best_f1 = -float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            batch_losses = []
            noise_std = noise_std_at_epoch(epoch, max_epochs, initial_noise_std, final_noise_std, mode=noise_mode)

            for xb_emb1, xb_emb2, yb in train_loader:
                xb_emb1 = xb_emb1.to(device)
                xb_emb2 = xb_emb2.to(device)
                yb = yb.to(device)

                if noise_std > 0:
                    xb_emb1 = xb_emb1 + torch.randn_like(xb_emb1) * noise_std
                    xb_emb2 = xb_emb2 + torch.randn_like(xb_emb2) * noise_std

                optimizer.zero_grad()
                logits = model(xb_emb1, xb_emb2)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_losses.append(np.mean(batch_losses))

            auc_v, aupr_v, f1_v, p_v, r_v, probs_v, preds_v, val_loss = evaluate(
                model, [X_emb1_test, X_emb2_test], y_test, device,
                threshold=0.5, return_loss=True, criterion=criterion
            )

            print(
                f"Epoch {epoch}: train_loss={train_losses[-1]:.4f} "
                f"val_loss={val_loss:.4f} AUC_val={auc_v:.4f} F1_val={f1_v:.4f}"
            )

            if f1_v > best_f1:
                best_f1 = f1_v
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Fold {fold}: Early stopping at epoch {epoch} (no F1 improvement for {patience} epochs)")
                    break

        if best_state:
            model.load_state_dict(best_state)

        auc, aupr, f1, precision, recall, y_prob_val, y_pred_val, _ = evaluate(
            model, [X_emb1_test, X_emb2_test], y_test, device
        )
        best_thr, best_p, best_r = find_best_threshold(y_test, y_prob_val)
        print(f"选择的最佳阈值: {best_thr:.3f}")

        # --------- IND test (test_dataset.csv) ----------
        # X_emb1_test_ind = scaler_emb1.transform(embs_test_ind[0])
        # X_emb2_test_ind = scaler_emb2.transform(embs_test_ind[1])

        auc, aupr, f1, precision, recall, y_prob, y_pred, _ = evaluate(
            model, [X_emb1_test, X_emb2_test], y_test, device, threshold=best_thr
        )

        print(f"TEST AUC: {auc:.3f}, AUPR: {aupr:.3f}, F1: {f1:.3f}, P: {precision:.3f}, R: {recall:.3f}")

        all_fold_probs.append(y_prob)

        # --------- 保存：模型、scaler、阈值 ----------
        ckpt = {
            "state_dict": model.state_dict(),
            "embedding1_dim": embedding1_dim,
            "embedding2_dim": embedding2_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "pre_model": pre_model,
        }
        save_dir = os.path.join("artifacts", f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(save_dir, f"fusion_mlp.pt"))

        joblib.dump(scaler_emb1, os.path.join(save_dir, f"scaler_emb1.pkl"))
        joblib.dump(scaler_emb2, os.path.join(save_dir, f"scaler_emb2.pkl"))

        with open(os.path.join(save_dir, "best_threshold.txt"), "w") as f:
            f.write(str(best_thr))

        # 保存 metrics
        all_metrics["AUC"].append(auc)
        all_metrics["AUPR"].append(aupr)
        all_metrics["F1"].append(f1)
        all_metrics["Precision"].append(precision)
        all_metrics["Recall"].append(recall)

    if len(all_fold_probs) > 0:
        print("\n===== 平均结果 =====")
        for m in all_metrics:
            print(f"{m}: {np.mean(all_metrics[m]):.4f} ± {np.std(all_metrics[m]):.4f}")


if __name__ == "__main__":
    main()
