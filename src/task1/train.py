from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from .data import build_flowers102_loaders
from .metrics import accuracy_topk
from .models import build_model
from .utils import (
    checkpoint_payload,
    ensure_dir,
    finish_tracker,
    get_device,
    load_config,
    log_metrics,
    maybe_init_tracker,
    resolve_run_dirs,
    save_json,
    set_seed,
    update_from_cli,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task1 Flowers102 classifiers.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional dotted overrides, e.g. train.epochs=5 model.pretrained=false",
    )
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download.")
    return parser.parse_args()


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    lr = float(train_cfg["learning_rate"])
    head_lr = train_cfg.get("head_learning_rate")
    backbone_lr = train_cfg.get("backbone_learning_rate")
    wd = float(train_cfg.get("weight_decay", 0.0))
    name = train_cfg.get("optimizer", "adamw").lower()
    if head_lr is not None or backbone_lr is not None:
        head_lr = float(head_lr if head_lr is not None else lr)
        backbone_lr = float(backbone_lr if backbone_lr is not None else lr)
        head_params = []
        backbone_params = []
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param_name.startswith("fc."):
                head_params.append(param)
            else:
                backbone_params.append(param)
        params = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    name = config["train"].get("scheduler", "cosine")
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["train"]["epochs"]),
            eta_min=float(config["train"]["learning_rate"]) * 0.01,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {name}")


def make_scaler(device: torch.device, enabled: bool):
    if not enabled or device.type == "cpu":
        return None
    try:
        return torch.amp.GradScaler(device.type, enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, enabled=True)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    grad_clip_norm: float | None,
    amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = labels.size(0)
        top1, top5 = accuracy_topk(logits.detach(), labels)
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, amp: bool, split: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, labels in tqdm(loader, desc=split, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        top1, top5 = accuracy_topk(logits, labels)
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


def write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_top1",
        "train_top5",
        "val_loss",
        "val_top1",
        "val_top5",
        "epoch_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict[str, float]], out_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_top1"] * 100 for row in rows], label="train")
    axes[1].plot(epochs, [row["val_top1"] * 100 for row in rows], label="val")
    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = update_from_cli(load_config(args.config), args.override)
    set_seed(int(config.get("seed", 42)))
    run_dir, ckpt_dir = resolve_run_dirs(config)
    save_json(config, run_dir / "config.json")

    device = get_device()
    train_cfg = config["train"]
    loaders = build_flowers102_loaders(
        root=config["data_root"],
        batch_size=int(train_cfg["batch_size"]),
        image_size=int(train_cfg["image_size"]),
        num_workers=int(train_cfg["num_workers"]),
        download=not args.no_download,
    )

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = make_scaler(device, bool(train_cfg.get("amp", True)))
    tracker = maybe_init_tracker(config, run_dir)

    best_val_top1 = 0.0
    best_epoch = 0
    rows: list[dict[str, float]] = []
    patience = int(train_cfg.get("early_stop_patience", 0))
    stale_epochs = 0
    epochs = int(train_cfg["epochs"])

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders.train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
            amp=bool(train_cfg.get("amp", True)),
        )
        val_metrics = evaluate(model, loaders.val, criterion, device, bool(train_cfg.get("amp", True)), split="val")
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "epoch_seconds": time.time() - start,
        }
        rows.append(row)
        write_metrics_csv(run_dir / "metrics.csv", rows)
        plot_curves(rows, run_dir / "curves.png")
        log_metrics(tracker, row, step=epoch)

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_val_top1,
            config=config,
        )
        torch.save(payload, ckpt_dir / "last.pt")

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            stale_epochs = 0
            payload["best_metric"] = best_val_top1
            torch.save(payload, ckpt_dir / "best.pt")
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"train_acc={row['train_top1']:.4f} val_acc={row['val_top1']:.4f}"
        )

        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best val top1={best_val_top1:.4f}")
            break

    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["model_state"])
    test_metrics = evaluate(model, loaders.test, criterion, device, bool(train_cfg.get("amp", True)), split="test")

    summary = {
        "experiment_name": config["experiment_name"],
        "device": device.type,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "test_loss": test_metrics["loss"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "weights": {
            "best": str(best_path),
            "last": str(ckpt_dir / "last.pt"),
        },
        "artifacts": {
            "metrics_csv": str(run_dir / "metrics.csv"),
            "curves_png": str(run_dir / "curves.png"),
            "config_json": str(run_dir / "config.json"),
        },
    }
    save_json(summary, run_dir / "summary.json")
    finish_tracker(tracker)
    print(f"Finished {config['experiment_name']}: test_top1={test_metrics['top1']:.4f}")


if __name__ == "__main__":
    main()
