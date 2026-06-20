"""
Dataset per Textual Inversion su SDXL.

Gestisce il caricamento delle immagini di riferimento e la tokenizzazione
per i due text encoder di SDXL (CLIP-L e OpenCLIP-G).
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Template di prompt per concetti "oggetto".
# Il placeholder {} viene sostituito con il token personalizzato (es. <cat2>).
OBJECT_TEMPLATES = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a good photo of a {}",
    "a rendition of the {}",
    "a rendition of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a depiction of a {}",
    "a depiction of the {}",
]

# Template alternativi per concetti di "stile".
STYLE_TEMPLATES = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a nice painting in the style of {}",
    "a dark painting in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a good painting in the style of {}",
    "a rendition in the style of {}",
    "a depiction in the style of {}",
]


class TextualInversionDataset(Dataset):
    """
    Dataset per il training di Textual Inversion su SDXL.

    Carica un piccolo set di immagini di riferimento (few-shot), le ripete
    `repeats` volte per riempire un'epoca e tokenizza ogni immagine con
    entrambi i tokenizer di SDXL.

    Args:
        data_root: Directory contenente le immagini di riferimento.
        tokenizer_one: CLIPTokenizer per CLIP-L (text_encoder).
        tokenizer_two: CLIPTokenizer per OpenCLIP-G (text_encoder_2).
        placeholder_token: Token speciale appreso, es. "<cat2>".
        learnable_property: "object" o "style" — determina il set di template.
        size: Risoluzione quadrata delle immagini (1024 per SDXL nativo).
        repeats: Numero di ripetizioni del dataset per epoch.
        center_crop: Se True usa un crop centrale invece di un resize diretto.
        flip_p: Probabilità di flip orizzontale casuale.
    """

    def __init__(
        self,
        data_root: str,
        tokenizer_one,
        tokenizer_two,
        placeholder_token: str,
        learnable_property: str = "object",
        size: int = 1024,
        repeats: int = 100,
        center_crop: bool = False,
        flip_p: float = 0.5,
    ):
        self.data_root = Path(data_root)
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.placeholder_token = placeholder_token
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.templates = (
            OBJECT_TEMPLATES if learnable_property == "object" else STYLE_TEMPLATES
        )

        # Raccoglie tutti i file immagine supportati
        self.image_paths = sorted(
            [
                p
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
                for p in self.data_root.glob(ext)
            ]
        )
        if not self.image_paths:
            raise ValueError(f"Nessuna immagine trovata in {self.data_root}")

        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats

        # Pipeline di trasformazione
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=flip_p),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # normalizza a [-1, 1]
            ]
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict:
        img_path = self.image_paths[index % self.num_images]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Scegli un template casuale e sostituisci il placeholder
        template = random.choice(self.templates)
        text = template.format(self.placeholder_token)

        # Tokenizza con entrambi i tokenizer
        input_ids_one = self._tokenize(self.tokenizer_one, text)
        input_ids_two = self._tokenize(self.tokenizer_two, text)

        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
        }

    def _tokenize(self, tokenizer, text: str) -> torch.Tensor:
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
