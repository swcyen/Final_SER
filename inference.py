import argparse
import torch
import yaml

from models.baseline_transformer import BaselineSERTransformer
from utils.audio import load_waveform


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict emotion for one wav file.")
    parser.add_argument("--audio", required=True, help="Path to a .wav file")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["paths"]["best_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = BaselineSERTransformer(
        num_classes=len(class_names),
        **config["model"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    waveform = load_waveform(
        args.audio,
        sample_rate=config["data"]["sample_rate"],
        duration_seconds=config["data"]["duration_seconds"],
    )  # [1, samples]
    waveform = waveform.unsqueeze(0).to(device)  # [batch=1, 1, samples]

    logits = model(waveform)  # [1, num_classes]
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    predicted_index = int(probabilities.argmax().item())

    print(f"Predicted emotion: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.4f}")
    for class_name, probability in zip(class_names, probabilities.cpu().tolist()):
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
