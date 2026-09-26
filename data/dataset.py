from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = {
    "houston": ("hsi", "aux1"),
    "muufl": ("hsi", "aux1"),
    "augsburg": ("hsi", "aux1", "aux2"),
}


class PatchDataset(Dataset):
    def __init__(self, root, dataset, split):
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.modalities = MODALITIES[dataset]
        split_file = self.root / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        self.ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _load(self, sample_id, name):
        return np.load(self.root / "patches" / f"{sample_id}_{name}.npy")

    def labels(self):
        return [int(self._load(sample_id, "label")) for sample_id in self.ids]

    def __getitem__(self, index):
        sample_id = self.ids[index]
        arrays = [self._load(sample_id, name) for name in self.modalities]
        if self.split == "train":
            if np.random.random() > 0.5:
                arrays = [np.flip(array, axis=0).copy() for array in arrays]
            if np.random.random() > 0.5:
                arrays = [np.flip(array, axis=1).copy() for array in arrays]
        tensors = []
        for array in arrays:
            if array.ndim == 2:
                array = array[..., None]
            tensors.append(torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float())
        label = torch.tensor(int(self._load(sample_id, "label")), dtype=torch.long)
        return (*tensors, label)


def build_dataset(name, root, split):
    if name not in MODALITIES:
        raise ValueError(f"Unsupported dataset: {name}")
    return PatchDataset(root, name, split)

