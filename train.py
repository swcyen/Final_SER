from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dataset import load_visec_datasets
from models.encoder import SpeechEncoder, count_parameters
from trainers.trainer import Trainer
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config("configs/config.yaml")
    set_seed(config["seed"])

    data_cfg = config["data"]
    train_cfg = config["training"]
    path_cfg = config["paths"]
    Path(path_cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(path_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, _, class_names = load_visec_datasets(config)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=pin_memory,
    )

    model = SpeechEncoder(
        num_classes=len(class_names),
        **config["model"],
    )
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_names=class_names,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
