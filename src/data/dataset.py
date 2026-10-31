"""
PyTorch Dataset and Grouped Leakage-Free Splitting Utilities.
"""
from __future__ import annotations

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


class MultimodalDataset(Dataset):
    """
    Multimodal PyTorch Dataset combining:
    - ECG tensor: [leads, samples]
    - Clinical tensor: [num_features]
    - Target: float scalar (0.0 or 1.0)
    - Metadata: record_id, patient_id
    """
    def __init__(
        self,
        ecg_tensors: Union[np.ndarray, torch.Tensor],
        clinical_tensors: Union[np.ndarray, torch.Tensor],
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        record_ids: Optional[List[str]] = None,
        patient_ids: Optional[List[str]] = None
    ):
        self.ecg = torch.as_tensor(ecg_tensors, dtype=torch.float32)
        self.clinical = torch.as_tensor(clinical_tensors, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.as_tensor(targets, dtype=torch.float32)
        else:
            self.targets = None
        self.record_ids = record_ids or [f"REC_{i}" for i in range(len(self.ecg))]
        self.patient_ids = patient_ids or [f"PAT_{i}" for i in range(len(self.ecg))]

        if len(self.ecg) != len(self.clinical):
            raise ValueError(f"ECG length ({len(self.ecg)}) does not match Clinical length ({len(self.clinical)})")
        if self.targets is not None and len(self.targets) != len(self.ecg):
            raise ValueError(f"Targets length ({len(self.targets)}) does not match ECG length ({len(self.ecg)})")

    def __len__(self) -> int:
        return len(self.ecg)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target = self.targets[idx] if self.targets is not None else torch.tensor(0.0)
        return self.ecg[idx], self.clinical[idx], target


def patient_level_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    target_col: str = "target",
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Strict patient-level holdout partition using GroupShuffleSplit.
    Guarantees 0 patient overlap across train, val, and test splits.
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    groups = df[patient_col].to_numpy()
    y = df[target_col].to_numpy() if target_col in df.columns else np.zeros(len(df))

    # Split off test set first
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter_test.split(df, y, groups))

    # Split train_val into train and val
    train_val_groups = groups[train_val_idx]
    train_val_y = y[train_val_idx]
    relative_val_size = val_size / (1.0 - test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    tr_rel, val_rel = next(splitter_val.split(train_val_idx, train_val_y, train_val_groups))

    train_idx = train_val_idx[tr_rel]
    val_idx = train_val_idx[val_rel]

    # Verify zero patient overlap assertion
    train_pts = set(groups[train_idx])
    val_pts = set(groups[val_idx])
    test_pts = set(groups[test_idx])

    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage detected between train and val"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage detected between train and test"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage detected between val and test"

    return train_idx, val_idx, test_idx


def create_dataloaders(
    train_dataset: MultimodalDataset,
    val_dataset: MultimodalDataset,
    test_dataset: Optional[MultimodalDataset] = None,
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    if test_dataset is not None:
        loaders["test"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loaders
