import hashlib
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from unimol_tools import UniMolRepr
from gensim.models import word2vec
from rdkit import Chem
from mol2vec.features import sentences2vec,  mol2alt_sentence


def featurize_with_chemberta(df, smiles_col="smiles", model_name="./model/ChemBERTa-zinc-base-v1", device="cuda"):
    """
    使用 ChemBERTa 将 SMILES 转换为 embedding
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for smi in tqdm(df[smiles_col], desc="Featurizing with ChemBERTa"):
            tokens = tokenizer(smi, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**tokens)

            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb.squeeze())

    X = np.array(embeddings)
    y = df["label"].values if "label" in df.columns else None
    return X, y


def featurize_with_mol2vec(df, smiles_col="smiles", model_path="model/model_300dim.pkl"):
    """
    使用 Mol2Vec 将 SMILES 转换为 embedding
    """
    w2v_model = word2vec.Word2Vec.load(model_path)

    embeddings = []
    for smi in tqdm(df[smiles_col], desc="Featurizing with Mol2Vec"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            vec = np.zeros(w2v_model.vector_size)
        else:
            sentence = mol2alt_sentence(mol, 1)
            vec = sentences2vec([sentence], w2v_model, unseen="UNK")[0]

        embeddings.append(vec)

    X = np.array(embeddings)
    y = df["label"].values if "label" in df.columns else None
    return X, y


def featurize_with_unimol(
        df,
        smiles_col="smiles",
        embedding_dim=768,
        cache_dir="./cache_unimol2"
):

    os.makedirs(cache_dir, exist_ok=True)
    smiles_list = df[smiles_col].tolist()
    data_hash = hashlib.md5((",".join(smiles_list)).encode()).hexdigest()[:10]
    cache_file = os.path.join(cache_dir, f"unimol2_emb_{data_hash}.npy")
    cache_label = os.path.join(cache_dir, f"unimol2_label_{data_hash}.npy")
    cache_valid_idx = os.path.join(cache_dir, f"unimol2_valididx_{data_hash}.npy")

    if os.path.exists(cache_file):
        print(f"已检测到 UniMol2 缓存文件，直接加载: {cache_file}")
        X = np.load(cache_file)
        y = np.load(cache_label) if os.path.exists(cache_label) else None

        if os.path.exists(cache_valid_idx):
            valid_idx = np.load(cache_valid_idx).tolist()
        else:
            valid_idx = list(range(X.shape[0]))

        return X, y, valid_idx

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/checkpoint.pt'))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"使用 UniMol2 提取特征 (device={device}) ...")

    repr_model = UniMolRepr(
        data_type='molecule',
        remove_hs=False,
        model_name='unimolv2',
        model_size='84m',
        pretrained_model_path=model_path,
        device=device
    )

    embeddings = []
    failed_smiles = []

    for smi in tqdm(smiles_list, desc="Featurizing with UniMol2"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Invalid SMILES")

            mol_repr = repr_model.get_repr([smi], return_atomic_reprs=False)
            if mol_repr is None or len(mol_repr) == 0:
                raise ValueError("Empty embedding")

            vec = mol_repr[0]
            if np.isnan(vec).any():
                raise ValueError("NaN embedding")

        except Exception as e:
            failed_smiles.append((smi, str(e)))
            vec = np.zeros(embedding_dim, dtype=np.float32)

        embeddings.append(vec)

    if failed_smiles:
        print(f"[Warning] {len(failed_smiles)} SMILES failed during featurization (showing up to 5):")
        for smi, err in failed_smiles[:5]:
            print(f"  - {smi}: {err}")

    valid_idx = [i for i, emb in enumerate(embeddings) if getattr(np.array(emb), "shape", (len(emb),))[0] == embedding_dim]
    invalid_count = len(embeddings) - len(valid_idx)

    embeddings = [embeddings[i] for i in valid_idx]
    X = np.vstack(embeddings).astype(np.float32)
    y = df.iloc[valid_idx]["label"].values.astype(np.float32) if "label" in df.columns else None

    np.save(cache_file, X)
    if y is not None:
        np.save(cache_label, y)
    np.save(cache_valid_idx, np.array(valid_idx, dtype=np.int32))

    print(f"UniMol2 特征提取完成并保存缓存: {cache_file} (valid {len(valid_idx)}/{len(smiles_list)})")
    return X, y, valid_idx