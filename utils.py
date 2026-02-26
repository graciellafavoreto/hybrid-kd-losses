"""
utils.py — Shared utilities for teachers and students
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from pathlib import Path
from config import BASE_PATH

class NoduleDataset(Dataset):
    """
    Loads .npy files with shape (224, 224, 3) in raw HU.
    Applies:
        - HU clipping and normalization to [0,1]
        - ImageNet normalization
        - Optional data augmentation (training only)
    """
    
    def __init__(self, df, base_path, augment=False):
        self.samples   = df.to_dict('records')
        self.base_path = Path(base_path)
        self.augment   = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        slices = np.load(
            self.base_path / s['patient_id'] / 
            f"nodule_{s['nodule_id']}_crop.npy"
        )  # (224, 224, 3) raw HU

        slices = np.clip(slices, -1000, 600)
        slices = (slices + 1000) / 1600.0

        # (H, W, C) → (C, H, W) 
        slices = np.transpose(slices, (2, 0, 1)).astype(np.float32)  # (3, 224, 224)

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        slices = (slices - mean) / std

        # Data augmentation (training only)
        if self.augment:
            if np.random.rand() > 0.5:
                slices = np.flip(slices, axis=1).copy()
            if np.random.rand() > 0.5:
                slices = np.flip(slices, axis=2).copy()
            if np.random.rand() > 0.5:
                slices = np.rot90(slices, k=np.random.randint(1, 4), axes=(1, 2)).copy()

        return (
            torch.from_numpy(slices).float(),
            torch.tensor(s['label'], dtype=torch.long),
        )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
        labels.extend(y.numpy())

    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        'acc'      : float((preds == labels).mean()),
        'auc'      : float(roc_auc_score(labels, probs)),
        'f1'       : float(f1_score(labels, preds, zero_division=0)),
        'sens'     : tp / (tp + fn) if (tp + fn) else 0.0,
        'spec'     : tn / (tn + fp) if (tn + fp) else 0.0,
        'confusion': (tn, fp, fn, tp),
    }

# Patient-Level Split
def patient_split(df, test_size=1/7, val_size=1/6, random_state=42):
    """
    Splits data at patient level (no leakage).

    Representative patient label:
        If a patient has at least one malignant nodule,
        the patient is labeled as malignant (1).
    """
    
    pat_label = (
        df.groupby('patient_id')['label']
        .max() 
    )
    
    patients = pat_label.index.values
    labels   = pat_label.values

    p_rem, p_test = train_test_split(
        patients, 
        test_size=test_size,
        stratify=labels, 
        random_state=random_state,
    )
    
    p_train, p_val = train_test_split(
        p_rem, 
        test_size=val_size,
        stratify=pat_label.loc[p_rem].values,
        random_state=random_state,
    )

    return (
        df[df['patient_id'].isin(p_train)].reset_index(drop=True),
        df[df['patient_id'].isin(p_val)].reset_index(drop=True),
        df[df['patient_id'].isin(p_test)].reset_index(drop=True),
    )

def assign_labels(df, benign_thresh=2.5, malign_thresh=3.5):
    """
    label=0 (benign) : if malignancy_median <= benign_thresh
    label=1 (malignant) : if malignancy_median >= malign_thresh
    label=-1 (ambiguous): excluded later
    """
    
    df = df.copy()
    df['label'] = -1
    
    df.loc[df['malignancy_median'] <= benign_thresh, 'label'] = 0
    df.loc[df['malignancy_median'] >= malign_thresh, 'label'] = 1
    
    return df