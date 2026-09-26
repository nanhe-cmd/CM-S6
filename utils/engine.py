import csv
import json
import math
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader

from data import build_dataset
from models import parameter_count


SCALAR_NAMES = {"alpha", "beta", "gamma", "eta", "zeta"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(value="auto"):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def compute_metrics(predictions, labels, num_classes):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    overall = float((predictions == labels).mean())
    per_class = []
    for class_id in range(num_classes):
        mask = labels == class_id
        score = float((predictions[mask] == labels[mask]).mean()) if mask.any() else 0.0
        per_class.append(score)
    return {
        "OA": 100.0 * overall,
        "AA": 100.0 * float(np.mean(per_class)),
        "Kappa": 100.0 * float(cohen_kappa_score(labels, predictions)),
        "per_class": [100.0 * value for value in per_class],
    }


def build_optimizer(model, config):
    training = config["training"]
    base_parameters = []
    scalar_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.rsplit(".", 1)[-1] in SCALAR_NAMES:
            scalar_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    groups = [
        {
            "params": base_parameters,
            "lr": training["learning_rate"],
            "weight_decay": training["weight_decay"],
        }
    ]
    if scalar_parameters:
        groups.append(
            {
                "params": scalar_parameters,
                "lr": training["learning_rate"] * training["scalar_lr_multiplier"],
                "weight_decay": training["weight_decay"],
            }
        )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, config, epochs):
    training = config["training"]
    minimum_ratio = max(training["minimum_lr"] / training["learning_rate"], 0.0)

    def factor(epoch):
        progress = min(max(epoch, 0), max(epochs, 1)) / max(epochs, 1)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def class_weighted_loss(dataset, config, device):
    counts = Counter(dataset.labels())
    num_classes = config["num_classes"]
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (max(counts.get(index, 1), 1) * num_classes) for index in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    weights = weights.clamp(max=config["training"]["class_weight_max"])
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=config["training"]["label_smoothing"],
    )


def _worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_loaders(config, seed, device):
    root = Path(config["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"Prepared dataset not found: {root}. Run data/prepare.py before training."
        )
    train_dataset = build_dataset(config["dataset"], root, "train")
    val_dataset = build_dataset(config["dataset"], root, "val")
    test_dataset = build_dataset(config["dataset"], root, "test")
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": config["training"]["workers"],
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_seed,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        **common,
    )
    return train_dataset, train_loader, val_loader, test_loader


def forward_batch(model, batch, device):
    inputs = [item.to(device, non_blocking=True) for item in batch[:-1]]
    labels = batch[-1].to(device, non_blocking=True)
    return model(*inputs), labels


def train_epoch(model, loader, criterion, optimizer, scaler, device, gradient_clip):
    model.train()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits, labels = forward_batch(model, batch, device)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.shape[0]
        total_samples += labels.shape[0]
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    predictions = []
    labels = []
    for batch in loader:
        logits, target = forward_batch(model, batch, device)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(target.cpu().tolist())
    return compute_metrics(predictions, labels, num_classes)


def _available_output(path):
    path = Path(path)
    if not path.exists() or not any(path.iterdir()):
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_training(
    config,
    model,
    run_name,
    run=1,
    seed=202601,
    device="auto",
    output=None,
    epochs=None,
    epoch_callback=None,
):
    set_seed(seed)
    device = resolve_device(device)
    epochs = int(epochs or config["training"]["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    preferred = Path(output) if output else Path(config["output_root"]) / run_name / f"run_{run:02d}_seed_{seed}"
    run_directory = _available_output(preferred.expanduser().resolve())
    run_directory.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    train_dataset, train_loader, val_loader, test_loader = build_loaders(config, seed, device)
    criterion = class_weighted_loss(train_dataset, config, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, epochs)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_key = None
    best_epoch = 0
    best_state = None
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            config["training"]["gradient_clip"],
        )
        scheduler.step()
        if epoch_callback is not None:
            epoch_callback(epoch, model, loss)

        val_metrics = evaluate(model, val_loader, device, config["num_classes"])
        row = {
            "epoch": epoch,
            "loss": loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_OA": val_metrics["OA"],
            "val_AA": val_metrics["AA"],
            "val_Kappa": val_metrics["Kappa"],
        }
        history.append(row)
        key = (val_metrics["OA"], val_metrics["AA"])
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d}/{epochs:03d} loss={loss:.4f} "
            f"val_OA={val_metrics['OA']:.2f} val_AA={val_metrics['AA']:.2f} "
            f"val_Kappa={val_metrics['Kappa']:.2f} best={best_key[0]:.2f}@{best_epoch}"
        )

    final_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, config["num_classes"])
    test_metrics["epoch"] = best_epoch
    elapsed = time.perf_counter() - start_time
    summary = {
        "dataset": config["dataset"],
        "run_name": run_name,
        "run": run,
        "seed": seed,
        "epochs": epochs,
        "device": str(device),
        "parameters": parameter_count(model),
        "parameters_M": parameter_count(model) / 1e6,
        "training_seconds": elapsed,
        "selection": {"metric": "validation OA, tie-break validation AA", "epoch": best_epoch,
                      "val_OA": best_key[0], "val_AA": best_key[1]},
        "best": test_metrics,
    }
    config_snapshot = {key: value for key, value in config.items() if key != "config_path"}
    with (run_directory / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2)
    with (run_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_csv(run_directory / "history.csv", history)
    torch.save(best_state, run_directory / "best.pt")
    torch.save(final_state, run_directory / "last.pt")
    print(
        f"test OA={test_metrics['OA']:.2f} AA={test_metrics['AA']:.2f} "
        f"Kappa={test_metrics['Kappa']:.2f} selected_epoch={best_epoch}"
    )
    print(f"results={run_directory}")
    return run_directory, summary
