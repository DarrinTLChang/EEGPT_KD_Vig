from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MyBinary5xTR2625Config:
    # root can be absolute OR relative to the working directory you run from
    root: str = "my_binary_5xtr_2625"
    x_name: str = "X.npy"
    y_name: str = "y.npy"


class MyBinary5xTR2625Dataset(Dataset):
    """
    Returns an 8-tuple to match the training script expectation:
      (fmri, eeg, physio, eeg_index_linear_raw, eeg_index_linear_smoothed,
       eeg_index_binary, alpha_theta_ratio, vigilance_seg)

    Only eeg and vigilance_seg are real; the rest are dummy scalars.
    """


    def __init__(self, dataset_config, split_set="fewshot_train", **kwargs):

        root = Path(dataset_config.root)

        # Directly use split_set as folder name
        split = split_set  # e.g. "fewshot_train", "fewshot_test"

        self.X = np.load(root / split / dataset_config.x_name).astype(np.float32)
        self.y = np.load(root / split / dataset_config.y_name).astype(np.int64)

        print(f"[MyBinary5xTR2625Dataset] split_set={split_set}")
        print(f"  X: {self.X.shape}  y: {self.y.shape}  pos_frac={self.y.mean():.3f}")

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.X[idx])                # (2625, 26)
        vigilance_seg = torch.tensor(self.y[idx]).long()   # scalar

        dummy = torch.zeros(1, dtype=torch.float32)

        return (
            dummy,          # fmri
            eeg,            # eeg
            dummy,          # physio
            dummy,          # eeg_index_linear_raw
            dummy,          # eeg_index_linear_smoothed
            dummy,          # eeg_index_binary
            dummy,          # alpha_theta_ratio
            vigilance_seg,  # label
        )