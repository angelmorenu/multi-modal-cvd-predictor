# src/eval.py
import os, numpy as np, torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from src.model import MultiModalCVD, load_checkpoint

PROC = "data/processed"
ART  = "artifacts"
ECG_LEN = 2000  # must match train

def load_tab(split):
    X = np.load(f"{PROC}/tabular_{split}_X.npy")
    y = np.load(f"{PROC}/tabular_{split}_y.npy").astype("int64")
    if X.ndim == 1: X = X[None, :]
    return X.astype("float32"), y

def load_ecg(split):
    p = f"{PROC}/ecg_{split}.npy"
    if not os.path.exists(p): return None
    E = np.load(p).astype("float32")
    if E.ndim == 1: E = E[None, :]
    # pad/crop
    if E.shape[1] < ECG_LEN:
        pad = np.zeros((E.shape[0], ECG_LEN - E.shape[1]), dtype="float32")
        E = np.concatenate([E, pad], axis=1)
    else:
        E = E[:, :ECG_LEN]
    return E

def main():
    Xt, yt = load_tab("test")
    Ecg    = load_ecg("test")
    if Ecg is None:  # tab-only fallback
        Ecg = np.zeros((Xt.shape[0], ECG_LEN), dtype="float32")

    # infer dims
    tab_dim = Xt.shape[1]
    model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)
    load_checkpoint(model, f"{ART}/model.pt", map_location="cpu")
    model.eval()

    with torch.inference_mode():
        tab = torch.from_numpy(Xt).float()
        ecg = torch.from_numpy(Ecg)[:, None, :].float()
        logits = model(tab, ecg)
        probs = torch.softmax(logits, dim=-1).numpy()[:, 1]
        preds = (probs >= 0.5).astype("int64")

    acc   = accuracy_score(yt, preds)
    try:
        roc  = roc_auc_score(yt, probs)
    except Exception:
        roc = float("nan")
    pr    = average_precision_score(yt, probs)
    brier = brier_score_loss(yt, probs)
    cm    = confusion_matrix(yt, preds)

    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {roc:.3f}")
    print(f"PR  AUC: {pr:.3f}")
    print(f"Brier  : {brier:.4f}")
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
