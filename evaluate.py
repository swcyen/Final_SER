from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dataset import load_visec_datasets
from models.baseline_transformer import BaselineSERTransformer
from utils.metrics import (
    compute_metrics,
    make_classification_report,
    save_confusion_matrix,
)
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@torch.no_grad()
def collect_predictions(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    for waveform, labels in data_loader:
        waveform = waveform.to(device)  # [batch, 1, samples]
        logits = model(waveform)  # [batch, num_classes]
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())

    return all_labels, all_preds


def main() -> None:
    config = load_config("configs/config.yaml")
    set_seed(config["seed"])

    data_cfg = config["data"]
    path_cfg = config["paths"]
    _, _, test_dataset, config_class_names = load_visec_datasets(config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(path_cfg["best_checkpoint"], map_location="cpu")
    class_names = checkpoint.get(
        "class_names",
        config_class_names,
    )

    model = BaselineSERTransformer(
        num_classes=len(class_names),
        **config["model"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    report = make_classification_report(y_true, y_pred, class_names)

    output_dir = Path(path_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = output_dir / "confusion_matrix.png"
    report_path = output_dir / "classification_report.txt"

    save_confusion_matrix(y_true, y_pred, class_names, str(confusion_path))
    report_path.write_text(report, encoding="utf-8")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(report)
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved classification report: {report_path}")


if __name__ == "__main__":
    main()
