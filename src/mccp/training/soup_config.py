"""Translate a resolved run config into Soup's YAML contract."""

from __future__ import annotations

from typing import Any

from mccp.domain.spec import ResolvedConfig


def soup_val_split(n_train: int, n_val: int) -> float:
    """Soup has no explicit validation-file field for local data; it splits
    the tail with `idx = int(len * (1 - val_split))`. We append val rows to
    train rows and choose a split that lands exactly on the boundary:
    len*(1-vs) = n_train + 0.5, so int() yields n_train regardless of float
    rounding."""
    total = n_train + n_val
    if n_val == 0:
        return 0.0
    vs = (n_val - 0.5) / total
    if not 0 < vs <= 0.5:
        raise ValueError(f"validation set too large for Soup's val_split (n_val={n_val}, total={total})")
    assert int(total * (1 - vs)) == n_train
    return vs


def build_soup_yaml(
    cfg: ResolvedConfig, *, n_train: int, n_val: int, data_file: str = "./data/train_and_val.jsonl"
) -> dict[str, Any]:
    hp = cfg.hyperparameters
    target = cfg.target_modules if isinstance(cfg.target_modules, list) else "all-linear"
    return {
        "base": cfg.base_model.id,  # replaced with the pinned local snapshot by the trainer entrypoint
        "task": cfg.method,
        "backend": cfg.soup.backend,
        "data": {
            "train": data_file,
            "format": "chatml",
            "val_split": soup_val_split(n_train, n_val),
            "max_length": hp.max_length,
            "train_on_responses_only": True,
            "mask_history": True,
        },
        "training": {
            "epochs": hp.n_epochs,
            "lr": hp.learning_rate,
            "batch_size": hp.per_device_batch_size,
            "gradient_accumulation_steps": hp.gradient_accumulation_steps,
            "warmup_ratio": hp.warmup_ratio,
            "quantization": "4bit" if cfg.finetuning_type == "qlora" else cfg.soup.quantization,
            "optimizer": cfg.soup.optimizer,
            "scheduler": cfg.soup.scheduler,
            "save_steps": cfg.soup.save_steps,
            "logging_steps": cfg.soup.logging_steps,
            "stream_layers": cfg.soup.stream_layers,
            "lora": {
                "r": hp.lora_r,
                "alpha": hp.lora_alpha,
                "dropout": hp.lora_dropout,
                "target_modules": target,
            },
        },
        "output": "./output",
    }
