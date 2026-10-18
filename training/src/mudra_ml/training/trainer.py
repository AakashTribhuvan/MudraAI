"""Experiment runner: train -> early-stop on val macro-F1 -> calibrate -> evaluate on test.

A checkpoint stores everything inference needs (spec "Mistake 7"): weights, class list,
model config, feature config (sequence length, velocity, schema version), feature
standardisation statistics and the fitted temperature.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mudra_ml.config import project_path
from mudra_ml.datasets.augment import AugmentConfig
from mudra_ml.datasets.isl_dataset import (
    BACKGROUND_LABEL,
    BackgroundConfig,
    FeatureConfig,
    ISLLandmarkDataset,
    load_classes,
)
from mudra_ml.evaluation.evaluate import evaluate_run, predict
from mudra_ml.evaluation.metrics import compute_metrics
from mudra_ml.models.sequence_model import ModelConfig, build_model, count_parameters

TRAIN_DEFAULTS = {
    "epochs": 150,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "warmup_epochs": 5,
    "patience": 30,
    "amp": True,
    "class_weights": False,
    "grad_clip": 1.0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1) / num_classes
    return torch.tensor(weights, dtype=torch.float32)


def init_from_checkpoint(model: nn.Module, path: Path) -> int:
    """Transfer learning: copy every weight with a matching name and shape except the classifier head.

    ``path`` is a run directory (uses best.pt) or a checkpoint file. Returns tensors copied.
    """
    ckpt_path = path / "best.pt" if path.is_dir() else path
    source = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
    target = model.state_dict()
    copied = {
        k: v for k, v in source.items() if not k.startswith("head.") and k in target and target[k].shape == v.shape
    }
    if not copied:
        raise ValueError(f"No compatible weights in {ckpt_path}; check feature and model settings match")
    target.update(copied)
    model.load_state_dict(target)
    return len(copied)


def run_experiment(cfg: dict, output_root: Path | None = None) -> dict:
    """Train and evaluate one configuration. Returns the summary dict (also written to disk)."""
    name = cfg["name"]
    seed = int(cfg.get("seed", 42))
    train_cfg = {**TRAIN_DEFAULTS, **cfg.get("train", {})}
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = project_path(output_root or cfg.get("output_root", "experiments/runs")) / f"{name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    split_dir = project_path(cfg["data"]["split_dir"])
    classes = load_classes(split_dir)
    background = BackgroundConfig.from_dict(cfg.get("background"))
    if background.prob > 0:
        classes = [*classes, BACKGROUND_LABEL]
    feature_cfg = FeatureConfig.from_dict(cfg.get("features"))
    augment = AugmentConfig.from_dict(cfg["augment"]) if cfg.get("augment") is not None else None

    train_ds = ISLLandmarkDataset(split_dir / "train.csv", classes, feature_cfg, augment, seed=seed, background=background)
    val_ds = ISLLandmarkDataset(split_dir / "val.csv", classes, feature_cfg)
    test_ds = ISLLandmarkDataset(split_dir / "test.csv", classes, feature_cfg)
    mean, std = train_ds.compute_stats()
    for ds in (train_ds, val_ds, test_ds):
        ds.set_stats(mean, std)

    bs = int(train_cfg["batch_size"])
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, pin_memory=pin)

    model_cfg = ModelConfig(input_dim=feature_cfg.feature_dim, num_classes=len(classes), **cfg.get("model", {}))
    model = build_model(model_cfg).to(device)
    n_params = count_parameters(model)
    if cfg.get("init_from"):
        loaded = init_from_checkpoint(model, project_path(cfg["init_from"]))
        print(f"[{name}] initialised {loaded} tensors from {cfg['init_from']} (classifier head trained from scratch)")

    weight = _class_weights(train_ds.labels, len(classes)).to(device) if train_cfg["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(train_cfg["label_smoothing"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(train_cfg["epochs"])
    warmup_steps = steps_per_epoch * int(train_cfg["warmup_epochs"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(str(run_dir / "tensorboard"))

    print(
        f"[{name}] device={device} params={n_params:,} classes={len(classes)} "
        f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} feature_dim={feature_cfg.feature_dim}"
    )

    def checkpoint(extra: dict | None = None) -> dict:
        return {
            "model_state": model.state_dict(),
            "model_config": model_cfg.to_dict(),
            "feature_config": feature_cfg.to_dict(),
            "classes": classes,
            "feature_mean": mean,
            "feature_std": std,
            "train_config": train_cfg,
            "experiment": name,
            **(extra or {}),
        }

    history_path = run_dir / "history.csv"
    best_f1, best_epoch, epochs_since_best = -1.0, -1, 0
    start = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as hf:
        hist = csv.writer(hf)
        hist.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "seconds"])
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            t0 = time.time()
            model.train()
            loss_sum, correct, seen = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x, y)  # labels only matter for a margin head
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                seen += len(y)

            val_logits, val_labels = predict(model, val_loader, device, use_amp)
            val_loss = float(nn.functional.cross_entropy(torch.tensor(val_logits), torch.tensor(val_labels)))
            val_m = compute_metrics(val_logits, val_labels)
            row = [
                epoch,
                loss_sum / seen,
                correct / seen,
                val_loss,
                val_m["accuracy"],
                val_m["macro_f1"],
                scheduler.get_last_lr()[0],
                time.time() - t0,
            ]
            hist.writerow([f"{v:.5f}" if isinstance(v, float) else v for v in row])
            hf.flush()
            for tag, value in zip(["loss/train", "acc/train", "loss/val", "acc/val", "f1/val"], row[1:6]):
                writer.add_scalar(tag, value, epoch)

            improved = val_m["macro_f1"] > best_f1
            if improved:
                best_f1, best_epoch, epochs_since_best = val_m["macro_f1"], epoch, 0
                torch.save(checkpoint({"epoch": epoch}), run_dir / "best.pt")
            else:
                epochs_since_best += 1
            if epoch == 1 or epoch % 10 == 0 or improved:
                print(
                    f"[{name}] epoch {epoch:3d} train_loss {row[1]:.3f} acc {row[2]:.3f} | "
                    f"val_loss {val_loss:.3f} acc {val_m['accuracy']:.3f} macroF1 {val_m['macro_f1']:.3f}"
                    f"{' *' if improved else ''}"
                )
            if epochs_since_best >= int(train_cfg["patience"]):
                print(f"[{name}] early stop at epoch {epoch} (best epoch {best_epoch})")
                break
    writer.close()
    train_seconds = time.time() - start

    # Reload best weights from disk, calibrate on validation, evaluate once on test.
    summary = evaluate_run(run_dir)
    summary.update({"params": n_params, "train_seconds": round(train_seconds, 1)})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    t = summary["test"]
    print(
        f"[{name}] TEST acc {t['accuracy']:.4f} top5 {t['top5_accuracy']:.4f} macroF1 {t['macro_f1']:.4f} "
        f"ECE {t['ece']:.3f} (T={summary['temperature']:.2f}, calibration "
        f"{'ok' if summary['calibration_reliable'] else 'unreliable: too few val errors'}) -> {run_dir}"
    )
    return summary
