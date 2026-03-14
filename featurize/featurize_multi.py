import numpy as np
from typing import Tuple

from featurize.featurize_nlp import featurize_with_chemberta, featurize_with_mol2vec, featurize_with_unimol


def featurize_multi_model(df,
                          smiles_col: str = "smiles",
                          models: list = ["Mol2Vec", "ChemBERTa"],
                          unimol_dim: int = 768) -> Tuple[list, np.ndarray]:

    embeddings = []
    y = None
    valid_indices = None

    # 处理每个指定的模型
    for model in models:
        if model == "ChemBERTa":
            out = featurize_with_chemberta(df, smiles_col=smiles_col)
            if isinstance(out, (tuple, list)):
                emb = out[0]
                if len(out) > 1 and y is None:
                    y = out[1]
            else:
                emb = out
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(len(df), -1)
            embeddings.append(emb)

        elif model == "UniMol2":
            emb, y_unimol, valid_idx = featurize_with_unimol(df, smiles_col=smiles_col, embedding_dim=unimol_dim)
            if valid_indices is None:
                valid_indices = valid_idx
                if y_unimol is not None and y is None:
                    y = y_unimol
            # 如果已经有有效索引，则需要对齐数据
            elif valid_indices is not None:
                # 这里简化处理，实际应用中可能需要更复杂的对齐逻辑
                pass
            embeddings.append(emb)

        else:
            out = featurize_with_mol2vec(df, smiles_col=smiles_col)
            if isinstance(out, (tuple, list)):
                emb = out[0]
                if len(out) > 1 and y is None:
                    y = out[1]
            else:
                emb = out
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(len(df), -1)
            embeddings.append(emb)

    if valid_indices is not None:
        adjusted_embeddings = []
        for emb in embeddings:
            if emb.shape[0] != len(valid_indices):
                adjusted_embeddings.append(emb[valid_indices])
            else:
                adjusted_embeddings.append(emb)
        embeddings = adjusted_embeddings

        if y is not None and len(y) != len(valid_indices):
            y = y[valid_indices]

    return embeddings, y

