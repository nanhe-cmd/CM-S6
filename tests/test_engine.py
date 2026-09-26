import json

import numpy as np

from models import build_model
from utils.engine import run_training, set_seed


def _write_split(root, split, count):
    identifiers = []
    for index in range(count):
        identifier = f"{split}_{index}_0"
        identifiers.append(identifier)
        np.save(root / "patches" / f"{identifier}_hsi.npy", np.random.rand(3, 3, 2).astype(np.float32))
        np.save(root / "patches" / f"{identifier}_aux1.npy", np.random.rand(3, 3, 1).astype(np.float32))
        np.save(root / "patches" / f"{identifier}_label.npy", np.asarray(index % 2, np.int64))
    (root / f"{split}.txt").write_text("\n".join(identifiers), encoding="utf-8")


def test_one_epoch_training(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "patches").mkdir(parents=True)
    _write_split(data_root, "train", 4)
    _write_split(data_root, "val", 4)
    _write_split(data_root, "test", 4)
    config = {
        "dataset": "houston",
        "data_root": str(data_root),
        "output_root": str(tmp_path / "outputs"),
        "num_classes": 2,
        "channels": [2, 1],
        "class_names": ["class 0", "class 1"],
        "model": {"dim": 4, "state_dim": 2, "layers": 1},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.0005,
            "scalar_lr_multiplier": 10.0,
            "weight_decay": 0.0001,
            "label_smoothing": 0.1,
            "class_weight_max": 5.0,
            "minimum_lr": 0.000001,
            "gradient_clip": 1.0,
            "workers": 0,
        },
    }
    set_seed(202601)
    model = build_model(config, "full")
    output, summary = run_training(
        config,
        model,
        run_name="smoke",
        run=1,
        seed=202601,
        device="cpu",
        output=tmp_path / "run",
        epochs=1,
    )
    assert summary["selection"]["epoch"] == 1
    assert summary["best"]["epoch"] == 1
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()
    assert json.loads((output / "summary.json").read_text())["dataset"] == "houston"
