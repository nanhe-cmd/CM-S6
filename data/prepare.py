import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

from data.spatial import (
    cross_split_overlap_report,
    select_spatial_splits,
    training_fit_mask,
)


DATASETS = {
    "houston": {"classes": 15, "modalities": ("hsi", "aux1"), "sar": None},
    "augsburg": {"classes": 7, "modalities": ("hsi", "aux1", "aux2"), "sar": "aux1"},
    "muufl": {"classes": 11, "modalities": ("hsi", "aux1"), "sar": None},
}


def load_raw(dataset, source):
    source = Path(source)
    if dataset == "houston":
        hsi = loadmat(source / "HSI.mat")["HSI"].astype(np.float32)[:, :, 4:-4]
        aux1 = loadmat(source / "LiDAR.mat")["LiDAR"].astype(np.float32)[..., None]
        labels = np.maximum(
            loadmat(source / "TRLabel.mat")["TRLabel"],
            loadmat(source / "TSLabel.mat")["TSLabel"],
        )
    elif dataset == "muufl":
        hsi = loadmat(source / "HSI.mat")["HSI"].astype(np.float32)
        aux1 = loadmat(source / "LiDAR.mat")["LiDAR"].astype(np.float32)
        if aux1.ndim == 2:
            aux1 = aux1[..., None]
        labels = np.maximum(
            loadmat(source / "muufl_tr.mat")["training_map"],
            loadmat(source / "muufl_ts.mat")["testing_map"],
        )
    else:
        hsi = loadmat(source / "data_HS_LR.mat")["data_HS_LR"].astype(np.float32)
        aux1 = loadmat(source / "data_SAR_HR.mat")["data_SAR_HR"].astype(np.float32)
        aux2 = loadmat(source / "data_DSM.mat")["data_DSM"].astype(np.float32)[..., None]
        labels = np.maximum(
            loadmat(source / "TrainImage.mat")["TrainImage"],
            loadmat(source / "TestImage.mat")["TestImage"],
        )

    arrays = {"hsi": hsi, "aux1": aux1}
    if dataset == "augsburg":
        arrays["aux2"] = aux2
    labels = np.asarray(labels, dtype=np.int64)
    for name, array in arrays.items():
        if array.shape[:2] != labels.shape:
            raise ValueError(f"{name} shape {array.shape[:2]} does not match labels {labels.shape}")
    return arrays, labels


def fit_minmax(array, fit_mask):
    values = array[fit_mask]
    low = np.nanmin(values, axis=0).astype(np.float32)
    high = np.nanmax(values, axis=0).astype(np.float32)
    scale = np.maximum(high - low, 1e-8)
    normalized = np.clip((array - low) / scale, 0.0, 1.0).astype(np.float32)
    return normalized, {"low": low.tolist(), "high": high.tolist(), "fit_pixels": int(values.shape[0])}


def fit_percentile_clip(array, fit_mask, lower=1.0, upper=99.0):
    values = array[fit_mask]
    low = np.nanpercentile(values, lower, axis=0).astype(np.float32)
    high = np.nanpercentile(values, upper, axis=0).astype(np.float32)
    high = np.maximum(high, low + 1e-8)
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)
    stats = {
        "lower_percentile": float(lower),
        "upper_percentile": float(upper),
        "low": low.tolist(),
        "high": high.tolist(),
        "fit_pixels": int(values.shape[0]),
    }
    return normalized, stats


def fit_hsi_pca(hsi, fit_mask, components, seed, pca_samples):
    height, width, bands = hsi.shape
    fit_values = hsi[fit_mask]
    rng = np.random.default_rng(seed)
    if len(fit_values) > pca_samples:
        selected = rng.choice(len(fit_values), size=pca_samples, replace=False)
        fit_values = fit_values[selected]
    pca = PCA(n_components=components, random_state=seed)
    pca.fit(fit_values)
    reduced = pca.transform(hsi.reshape(-1, bands)).reshape(height, width, components)
    info = {
        "components": int(components),
        "fit_pixels_used": int(len(fit_values)),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return reduced.astype(np.float32), info


def pad_array(array, patch_size):
    pad = patch_size // 2
    return np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def save_patches(output, splits, processed, labels, patch_size):
    patch_dir = output / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    padded = {name: pad_array(array, patch_size) for name, array in processed.items()}
    coordinates = []
    class_counts = {}
    for split in ("train", "val", "test"):
        identifiers = []
        counts = {}
        for row, col, label in splits[split]:
            sample_id = f"{split}_{row}_{col}"
            identifiers.append(sample_id)
            counts[str(label - 1)] = counts.get(str(label - 1), 0) + 1
            coordinates.append((split, sample_id, row, col, int(label - 1)))
            for name, array in padded.items():
                patch = np.ascontiguousarray(array[row : row + patch_size, col : col + patch_size])
                np.save(patch_dir / f"{sample_id}_{name}.npy", patch.astype(np.float32))
            np.save(
                patch_dir / f"{sample_id}_label.npy",
                np.asarray(labels[row, col] - 1, dtype=np.int64),
            )
        (output / f"{split}.txt").write_text("\n".join(identifiers) + "\n", encoding="utf-8")
        class_counts[split] = counts
    with (output / "coordinates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "sample_id", "row", "col", "label_zero_based"))
        writer.writerows(coordinates)
    return class_counts


def prepare(dataset, source, output, patch_size, seed, train_per_class, val_per_class, pca_samples):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    arrays, label_map = load_raw(dataset, source)
    height, width = label_map.shape
    splits, diagnostics = select_spatial_splits(
        label_map,
        patch_size=patch_size,
        seed=seed,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
    )
    overlap = cross_split_overlap_report(splits, height, width, patch_size)
    if not overlap["pass"]:
        raise AssertionError(f"Cross-split patch overlap detected: {overlap}")
    fit_mask = training_fit_mask(splits["train"], height, width, patch_size)

    hsi, hsi_stats = fit_minmax(arrays["hsi"], fit_mask)
    hsi, pca_info = fit_hsi_pca(hsi, fit_mask, 20, seed, pca_samples)
    processed = {"hsi": hsi}
    preprocessing = {"hsi": {"normalization": hsi_stats, "pca": pca_info}}
    sar = DATASETS[dataset]["sar"]
    for name in DATASETS[dataset]["modalities"]:
        if name == "hsi":
            continue
        if name == sar:
            normalized, stats = fit_percentile_clip(arrays[name], fit_mask)
        else:
            normalized, stats = fit_minmax(arrays[name], fit_mask)
        processed[name] = normalized
        preprocessing[name] = stats

    class_counts = save_patches(output, splits, processed, label_map, patch_size)
    protocol = {
        "dataset": dataset,
        "patch_size": patch_size,
        "padding": "reflect",
        "seed": seed,
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "cross_split_min_chebyshev_distance": patch_size,
        "sample_counts": {split: len(splits[split]) for split in ("train", "val", "test")},
        "class_counts_zero_based": class_counts,
        "cross_split_overlap": overlap["pair_overlap_counts"],
        "class_shortfalls": diagnostics["class_shortfalls"],
        "dropped_test_near_existing_split": diagnostics["dropped_test_near_existing_split"],
        "preprocessing_fit_scope": "union of training patch windows",
        "preprocessing": preprocessing,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol["sample_counts"], indent=2))
    print(f"protocol={output / 'protocol.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--pca-samples", type=int, default=100000)
    args = parser.parse_args()
    prepare(
        args.dataset,
        args.source,
        args.output,
        args.patch_size,
        args.seed,
        args.train_per_class,
        args.val_per_class,
        args.pca_samples,
    )


if __name__ == "__main__":
    main()
