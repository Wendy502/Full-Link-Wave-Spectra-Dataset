import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class SARWaveHsDataset(Dataset):
    """
    Dataset for SAR-to-wave-height regression.

    Each .mat file contains:
      - Wave_Spectra          (256, 256)
      - SAR_Spectra           (256, 256)
      - incidenceAngle_Matrix (256, 256)
      - Hs                    (1, 1) — optional; returns 0.0 if missing (inference mode)
    """

    def __init__(self, folder_path, transform=None):
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform

        # Collect all .mat files
        self.sample_files = []
        for f in os.listdir(folder_path):
            if f.endswith('.mat'):
                self.sample_files.append(os.path.join(folder_path, f))
        self.sample_files.sort()

        # Pre-scan: check which files have the Hs field
        self.has_hs = []
        for mat_path in self.sample_files:
            with h5py.File(mat_path, 'r') as f:
                self.has_hs.append('Hs' in f)

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        mat_path = self.sample_files[idx]

        with h5py.File(mat_path, 'r') as f:
            # h5py loads in (W, H) — transpose to (H, W)
            wave = np.array(f['Wave_Spectra']).astype(np.float32).T
            sar  = np.array(f['SAR_Spectra']).astype(np.float32).T
            inc  = np.array(f['incidenceAngle_Matrix']).astype(np.float32).T

            # Hs may be absent in inference
            if 'Hs' in f:
                hs = np.array(f['Hs']).astype(np.float32).squeeze()
            else:
                hs = np.float32(0.0)

            # Normalization
            sar = np.log10(sar + 1) / 5
            inc = (inc - 20) / (40 - 20)

        # Add channel dimension: [C, H, W]
        if wave.ndim == 2:
            wave = wave[np.newaxis, :, :]
        if sar.ndim == 2:
            sar = sar[np.newaxis, :, :]
        if inc.ndim == 2:
            inc = inc[np.newaxis, :, :]

        wave = torch.from_numpy(wave)
        sar  = torch.from_numpy(sar)
        inc  = torch.from_numpy(inc)
        hs   = torch.tensor(hs, dtype=torch.float32)

        if self.transform:
            wave = self.transform(wave)
            sar  = self.transform(sar)
            inc  = self.transform(inc)

        return wave, sar, inc, hs
