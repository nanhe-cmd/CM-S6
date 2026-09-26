import numpy as np

from data import PatchDataset


def test_patch_dataset(tmp_path):
    patch_directory = tmp_path / "patches"
    patch_directory.mkdir()
    identifiers = ("train_0_0", "train_1_1")
    (tmp_path / "train.txt").write_text("\n".join(identifiers), encoding="utf-8")
    for index, identifier in enumerate(identifiers):
        np.save(patch_directory / f"{identifier}_hsi.npy", np.zeros((3, 3, 20), np.float32))
        np.save(patch_directory / f"{identifier}_aux1.npy", np.zeros((3, 3, 1), np.float32))
        np.save(patch_directory / f"{identifier}_label.npy", np.asarray(index, np.int64))

    dataset = PatchDataset(tmp_path, "houston", "train")
    hsi, lidar, label = dataset[0]
    assert hsi.shape == (20, 3, 3)
    assert lidar.shape == (1, 3, 3)
    assert label.ndim == 0
    assert dataset.labels() == [0, 1]
