import random
from copy import deepcopy

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdChemReactions
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from scipy.stats import ks_2samp, entropy
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedShuffleSplit, StratifiedGroupKFold, \
    StratifiedKFold
from collections import defaultdict
from imblearn.over_sampling import SMOTE


def evaluate_feature_independence(X_train, X_val, X_test, name="Feature", bandwidth=1.0):
    """
    计算 train/val/test 特征独立性指标，包括：
    - KS检验
    - 均值/方差差异
    - KL divergence
    - JS divergence
    - Maximum Mean Discrepancy (MMD)

    X_train, X_val, X_test: np.array, shape [n_samples, n_features]
    bandwidth: float, RBF kernel bandwidth for MMD
    """

    def compute_metrics(a, b):
        # 逐维 KS 检验
        ks_pvals = [ks_2samp(a[:, i], b[:, i]).pvalue for i in range(a.shape[1])]
        mean_diff = np.abs(a.mean() - b.mean())
        var_diff = np.abs(a.var() - b.var())

        # KL / JS divergence（逐维，做平均）
        # 首先将每列做 histogram（概率分布）
        kl_list, js_list = [], []
        bins = 50  # 可以调节
        for i in range(a.shape[1]):
            hist_a, _ = np.histogram(a[:, i], bins=bins, range=(a[:, i].min(), a[:, i].max()), density=True)
            hist_b, _ = np.histogram(b[:, i], bins=bins, range=(a[:, i].min(), a[:, i].max()), density=True)
            # 避免 0
            hist_a += 1e-8
            hist_b += 1e-8
            kl_list.append(entropy(hist_a, hist_b))  # KL(a||b)
            m = 0.5 * (hist_a + hist_b)
            js_list.append(0.5 * (entropy(hist_a, m) + entropy(hist_b, m)))  # JS

        kl_mean = np.mean(kl_list)
        js_mean = np.mean(js_list)

        # MMD (RBF kernel)
        K_aa = rbf_kernel(a, a, gamma=1.0 / (2 * bandwidth ** 2))
        K_bb = rbf_kernel(b, b, gamma=1.0 / (2 * bandwidth ** 2))
        K_ab = rbf_kernel(a, b, gamma=1.0 / (2 * bandwidth ** 2))
        mmd = K_aa.mean() + K_bb.mean() - 2 * K_ab.mean()

        return {
            "ks_mean_pval": np.mean(ks_pvals),
            "ks_min_pval": np.min(ks_pvals),
            "mean_diff": mean_diff,
            "var_diff": var_diff,
            "kl_mean": kl_mean,
            "js_mean": js_mean,
            "mmd": mmd
        }

    print(f"[{name}] train vs val:", compute_metrics(X_train, X_val))
    print(f"[{name}] train vs test:", compute_metrics(X_train, X_test))
    print(f"[{name}] val vs test:", compute_metrics(X_val, X_test))


def get_scaffold(smiles):
    """获取分子骨架（scaffold）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


# ========== 参数 ==========
TARGET_MIN = 8
TARGET_MAX = 12
SMILES_ENUM_N = 6
SEED = 42
random.seed(SEED)


# ========== 工具函数 ==========
def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def group_stratified_kfold_split(df, n_splits=5, seed=42):
    smiles_list = df["smiles"].tolist()
    y = np.array(df["label"].tolist())

    scaffolds = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffolds.append("NO_SCAFFOLD")
            else:
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            scaffolds.append("NO_SCAFFOLD")

    df["scaffold"] = scaffolds

    scaffold_to_indices = defaultdict(list)
    for i, scaf in enumerate(scaffolds):
        scaffold_to_indices[scaf].append(i)

    scaffold_to_indices = {k: v for k, v in scaffold_to_indices.items() if len(v) > 0}

    groups = list(scaffold_to_indices.keys())

    group_labels = []
    for scaf in groups:
        labels = [y[i] for i in scaffold_to_indices[scaf]]
        if len(labels) == 0:
            continue
        majority_label = int(np.round(np.mean(labels)))
        group_labels.append(majority_label)

    if len(groups) != len(group_labels):
        min_len = min(len(groups), len(group_labels))
        groups = groups[:min_len]
        group_labels = group_labels[:min_len]

    print(f"共有 {len(groups)} 个 scaffold 分组参与划分。")

    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_folds = []

    for fold, (train_groups_idx, test_groups_idx) in enumerate(skf_outer.split(groups, group_labels)):
        train_groups = [groups[i] for i in train_groups_idx]
        test_groups = [groups[i] for i in test_groups_idx]

        train_index = [i for g in train_groups for i in scaffold_to_indices[g]]
        test_index = [i for g in test_groups for i in scaffold_to_indices[g]]

        y_test = y[test_index]
        if len(np.unique(y_test)) < 2 or len(y_test) < 2:
            split_point = len(test_index) // 2
            val_index = test_index[:split_point]
            test_index = test_index[split_point:]
        else:
            skf_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            val_idx, test_idx = next(skf_inner.split(np.zeros(len(y_test)), y_test))
            val_index = [test_index[i] for i in val_idx]
            test_index = [test_index[i] for i in test_idx]

        train_index = train_index + val_index

        all_folds.append((train_index, test_index))

    return all_folds
