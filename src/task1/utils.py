from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_run_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    name = config["experiment_name"]
    run_dir = ensure_dir(Path(config["output_dir"]) / name)
    ckpt_dir = ensure_dir(Path(config["checkpoint_dir"]) / name)
    return run_dir, ckpt_dir


def update_from_cli(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted key=value overrides, e.g. train.epochs=5."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = yaml.safe_load(value)
    return config


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "config": config,
    }


def maybe_init_tracker(config: dict[str, Any], run_dir: Path):
    backend = config.get("logging", {}).get("backend", "none")
    if backend == "none":
        return None

    if backend == "wandb":
        import wandb

        kwargs = {
            "project": config["logging"].get("project", "dl-spatial-hw2-task1"),
            "name": config["experiment_name"],
            "config": config,
            "dir": str(run_dir),
        }
        if config["logging"].get("mode") is not None:
            kwargs["mode"] = config["logging"]["mode"]
        return wandb.init(**kwargs)

    if backend == "swanlab":
        import swanlab

        kwargs = {
            "project": config["logging"].get("project", "dl-spatial-hw2-task1"),
            "experiment_name": config["experiment_name"],
            "config": config,
            "logdir": str(run_dir),
        }
        if config["logging"].get("mode") is not None:
            kwargs["mode"] = config["logging"]["mode"]
        return swanlab.init(**kwargs)

    raise ValueError(f"Unknown logging backend: {backend}")


def log_metrics(tracker: Any, metrics: dict[str, float], step: int) -> None:
    if tracker is None:
        return
    if hasattr(tracker, "log"):
        tracker.log(metrics, step=step)
    else:
        import swanlab

        swanlab.log(metrics, step=step)


def finish_tracker(tracker: Any) -> None:
    if tracker is None:
        return
    if hasattr(tracker, "finish"):
        tracker.finish()
