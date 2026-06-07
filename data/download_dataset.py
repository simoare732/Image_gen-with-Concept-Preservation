"""
Scarica il dataset DreamBooth cat2 da HuggingFace e salva le immagini in data/instance_images/.
Utilizzo: python data/download_dataset.py
"""

from datasets import load_dataset
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "instance_images"
SAVE_DIR.mkdir(exist_ok=True)

print("Scaricamento dataset google/dreambooth (cat2)...")
ds = load_dataset("google/dreambooth", "cat2", split="train")

for i, sample in enumerate(ds):
    img = sample["image"]  # PIL Image
    img.save(SAVE_DIR / f"{i:03d}.jpg")

print(f"Salvate {len(ds)} immagini in {SAVE_DIR}")
