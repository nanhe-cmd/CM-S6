from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def _cell(coord, cell_size):
    return coord[0] // cell_size, coord[1] // cell_size


class _SpatialIndex:
    def __init__(self, distance):
        self.distance = int(distance)
        self.cell_size = max(1, self.distance + 1)
        self.cells = defaultdict(list)

    def add(self, coord):
        self.cells[_cell(coord, self.cell_size)].append(coord)

    def conflicts(self, coord):
        cell_r, cell_c = _cell(coord, self.cell_size)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for other in self.cells.get((cell_r + dr, cell_c + dc), ()):
                    if max(abs(coord[0] - other[0]), abs(coord[1] - other[1])) <= self.distance:
                        return True
        return False


def _spread_order(indices, coords, rng, block_size):
    blocks = defaultdict(list)
    for index in indices:
        row, col = (int(value) for value in coords[index])
        blocks[(row // block_size, col // block_size)].append(int(index))
    block_keys = list(blocks)
    rng.shuffle(block_keys)
    for key in block_keys:
        rng.shuffle(blocks[key])
    ordered = []
    while block_keys:
        next_keys = []
        for key in block_keys:
            values = blocks[key]
            if values:
                ordered.append(values.pop())
            if values:
                next_keys.append(key)
        block_keys = next_keys
    return ordered


def select_spatial_splits(
    label_map,
    patch_size=11,
    seed=2026,
    train_per_class=20,
    val_per_class=20,
    max_test_per_class=0,
):
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be odd and at least 3")
    if train_per_class < 1 or val_per_class < 1:
        raise ValueError("train_per_class and val_per_class must be positive")
    if max_test_per_class < 0:
        raise ValueError("max_test_per_class must be non-negative")

    label_map = np.asarray(label_map)
    rows, cols = np.where(label_map > 0)
    labels = label_map[rows, cols].astype(np.int64)
    coords = np.stack((rows, cols), axis=1)
    classes = sorted(int(value) for value in np.unique(labels))
    rng = np.random.default_rng(seed)
    distance = patch_size - 1
    block_size = patch_size
    train_index = _SpatialIndex(distance)
    val_index = _SpatialIndex(distance)
    test_index = _SpatialIndex(distance)
    chosen = {"train": [], "val": [], "test": []}
    used = set()
    diagnostics = {
        "distance_rule": f"Chebyshev center distance >= {distance + 1}",
        "patch_size": patch_size,
        "block_size": block_size,
        "seed": seed,
        "requested_per_class": {"train": train_per_class, "val": val_per_class},
        "dropped_test_near_existing_split": 0,
        "class_shortfalls": {"train": {}, "val": {}, "test": {}},
    }

    by_class = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)

    def conflicts_with_other_splits(coord, own):
        indexes = {"train": train_index, "val": val_index, "test": test_index}
        return any(index.conflicts(coord) for name, index in indexes.items() if name != own)

    class_order = list(classes)
    rng.shuffle(class_order)
    for class_id in class_order:
        candidates = _spread_order(by_class[class_id], coords, rng, block_size)
        best_pair = None
        probe = candidates[: min(len(candidates), 600)]
        probe_coords = coords[np.asarray(probe, dtype=np.int64)]
        if len(probe):
            distances = np.maximum(
                np.abs(probe_coords[:, None, 0] - probe_coords[None, :, 0]),
                np.abs(probe_coords[:, None, 1] - probe_coords[None, :, 1]),
            )
            order = np.dstack(np.unravel_index(np.argsort(distances.ravel())[::-1], distances.shape))[0]
            for val_pos, test_pos in order:
                val_candidate = probe[int(val_pos)]
                test_candidate = probe[int(test_pos)]
                if val_candidate == test_candidate or val_candidate in used or test_candidate in used:
                    continue
                val_coord = (int(coords[val_candidate, 0]), int(coords[val_candidate, 1]))
                test_coord = (int(coords[test_candidate, 0]), int(coords[test_candidate, 1]))
                if conflicts_with_other_splits(val_coord, "val"):
                    continue
                if conflicts_with_other_splits(test_coord, "test"):
                    continue
                separation = max(
                    abs(val_coord[0] - test_coord[0]),
                    abs(val_coord[1] - test_coord[1]),
                )
                if separation > distance:
                    best_pair = (val_candidate, test_candidate)
                    break
        if best_pair is None:
            diagnostics["class_shortfalls"]["val"][str(class_id)] = 1
            diagnostics["class_shortfalls"]["test"][str(class_id)] = 1
            continue
        val_candidate, test_candidate = best_pair
        val_coord = (int(coords[val_candidate, 0]), int(coords[val_candidate, 1]))
        test_coord = (int(coords[test_candidate, 0]), int(coords[test_candidate, 1]))
        chosen["val"].append((val_coord[0], val_coord[1], class_id))
        chosen["test"].append((test_coord[0], test_coord[1], class_id))
        val_index.add(val_coord)
        test_index.add(test_coord)
        used.update((val_candidate, test_candidate))

    for class_id in classes:
        candidates = _spread_order(by_class[class_id], coords, rng, block_size)
        picked = 0
        for index in candidates:
            if index in used:
                continue
            coord = (int(coords[index, 0]), int(coords[index, 1]))
            if conflicts_with_other_splits(coord, "train"):
                continue
            chosen["train"].append((coord[0], coord[1], class_id))
            train_index.add(coord)
            used.add(index)
            picked += 1
            if picked >= train_per_class:
                break
        if picked < train_per_class:
            diagnostics["class_shortfalls"]["train"][str(class_id)] = train_per_class - picked

    for class_id in classes:
        current = sum(item[2] == class_id for item in chosen["val"])
        candidates = _spread_order(by_class[class_id], coords, rng, block_size)
        for index in candidates:
            if current >= val_per_class or index in used:
                continue
            coord = (int(coords[index, 0]), int(coords[index, 1]))
            if conflicts_with_other_splits(coord, "val"):
                continue
            chosen["val"].append((coord[0], coord[1], class_id))
            val_index.add(coord)
            used.add(index)
            current += 1
        if current < val_per_class:
            diagnostics["class_shortfalls"]["val"][str(class_id)] = val_per_class - current

    for class_id in classes:
        current = sum(item[2] == class_id for item in chosen["test"])
        candidates = _spread_order(by_class[class_id], coords, rng, block_size)
        for index in candidates:
            if index in used:
                continue
            if max_test_per_class and current >= max_test_per_class:
                break
            coord = (int(coords[index, 0]), int(coords[index, 1]))
            if conflicts_with_other_splits(coord, "test"):
                diagnostics["dropped_test_near_existing_split"] += 1
                continue
            chosen["test"].append((coord[0], coord[1], class_id))
            test_index.add(coord)
            used.add(index)
            current += 1
        if current == 0:
            diagnostics["class_shortfalls"]["test"][str(class_id)] = 1

    for split in chosen:
        chosen[split].sort(key=lambda item: (item[0], item[1], item[2]))
    fatal_missing = {
        split: sorted(
            str(class_id)
            for class_id in classes
            if not any(item[2] == class_id for item in chosen[split])
        )
        for split in ("train", "val", "test")
    }
    if any(fatal_missing.values()):
        diagnostics["missing_classes"] = fatal_missing
        raise ValueError(f"Spatial split has a class with no samples: {fatal_missing}")
    diagnostics["counts"] = {split: len(values) for split, values in chosen.items()}
    diagnostics["class_counts"] = {
        split: {str(class_id): sum(item[2] == class_id for item in values) for class_id in classes}
        for split, values in chosen.items()
    }
    return chosen, diagnostics


def patch_window(row, col, patch_size, height, width):
    pad = patch_size // 2
    return (
        max(0, row - pad),
        min(height, row + pad + 1),
        max(0, col - pad),
        min(width, col + pad + 1),
    )


def _windows_overlap(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return ar0 < br1 and br0 < ar1 and ac0 < bc1 and bc0 < ac1


def cross_split_overlap_report(splits, height, width, patch_size):
    windows = {
        split: [patch_window(row, col, patch_size, height, width) for row, col, _ in values]
        for split, values in splits.items()
    }
    pairs = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        count = 0
        for window_a in windows[left]:
            for window_b in windows[right]:
                if _windows_overlap(window_a, window_b):
                    count += 1
        pairs[f"{left}_{right}"] = count
    return {"pair_overlap_counts": pairs, "pass": all(value == 0 for value in pairs.values())}


def training_fit_mask(train_centres, height, width, patch_size):
    mask = np.zeros((height, width), dtype=bool)
    for row, col, _ in train_centres:
        r0, r1, c0, c1 = patch_window(row, col, patch_size, height, width)
        mask[r0:r1, c0:c1] = True
    if not np.any(mask):
        raise ValueError("Training centres produce an empty fitting mask")
    return mask
