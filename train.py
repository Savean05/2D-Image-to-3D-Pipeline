import sys
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW

from tsr.system import TSR

PROJECT_ROOT = Path(__file__).resolve().parent
TRIPOSR_DIR = PROJECT_ROOT / "TripoSR"
if str(TRIPOSR_DIR) not in sys.path:
    sys.path.insert(0, str(TRIPOSR_DIR))


def train_model() -> None:
    """Fine-tune TripoSR tokenizer on custom dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to(device)
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    print("Unfreezing tokenizer parameters for fine-tuning...")
    for param in model.tokenizer.parameters():
        param.requires_grad = True

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=1e-5,
    )
    
    dataset_dir = PROJECT_ROOT / "dataset"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    images = sorted(path for path in dataset_dir.iterdir() if path.suffix.lower() == ".png")
    if not images:
        raise FileNotFoundError(f"No PNG images found in {dataset_dir}")

    print(f"Found {len(images)} images for training\n")
    
    for epoch in range(5):
        for image_path in images:
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
            
            optimizer.zero_grad()
            scene_codes = model(image, device=device)
            loss = scene_codes.mean() * 0.1
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch + 1}/{5} | {image_path.name} | Loss: {loss.item():.4f}")

    output_path = PROJECT_ROOT / "my_custom_triposr.ckpt"
    torch.save(model.state_dict(), output_path)
    print(f"\n✓ Saved custom weights to {output_path}")


if __name__ == "__main__":
    train_model()
