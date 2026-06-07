"""
Utility per salvare e caricare gli embedding appresi durante Textual Inversion su SDXL.

SDXL ha due text encoder:
  - text_encoder  (CLIP-L):    embedding dim 768  → chiave "clip_l"
  - text_encoder_2 (OpenCLIP-G): embedding dim 1280 → chiave "clip_g"

Il file .safetensors contiene entrambi i vettori e il nome del token.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_learned_embeddings(
    text_encoder_one,
    text_encoder_two,
    tokenizer_one,
    tokenizer_two,
    placeholder_token: str,
    save_path: str,
) -> None:
    """
    Salva i vettori embedding appresi per il placeholder_token in un file .safetensors.

    Args:
        text_encoder_one: CLIPTextModel (CLIP-L).
        text_encoder_two: CLIPTextModelWithProjection (OpenCLIP-G).
        tokenizer_one: CLIPTokenizer associato a text_encoder_one.
        tokenizer_two: CLIPTokenizer associato a text_encoder_two.
        placeholder_token: Token speciale, es. "<cat2>".
        save_path: Percorso del file di output.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    id1 = tokenizer_one.convert_tokens_to_ids(placeholder_token)
    id2 = tokenizer_two.convert_tokens_to_ids(placeholder_token)

    embed_one = (
        text_encoder_one.get_input_embeddings().weight[id1].detach().cpu()
    )  # shape [768]
    embed_two = (
        text_encoder_two.get_input_embeddings().weight[id2].detach().cpu()
    )  # shape [1280]

    save_file({"clip_l": embed_one, "clip_g": embed_two}, str(save_path))
    print(f"Embedding salvato in {save_path}")


def load_learned_embeddings(
    text_encoder_one,
    text_encoder_two,
    tokenizer_one,
    tokenizer_two,
    placeholder_token: str,
    load_path: str,
    device: str = "cpu",
) -> None:
    """
    Carica gli embedding salvati e li inietta nelle tabelle dei text encoder.

    I token devono essere già stati aggiunti ai tokenizer e le embedding table
    devono essere già state ridimensionate prima di chiamare questa funzione.

    Args:
        text_encoder_one: CLIPTextModel (CLIP-L).
        text_encoder_two: CLIPTextModelWithProjection (OpenCLIP-G).
        tokenizer_one: CLIPTokenizer associato a text_encoder_one.
        tokenizer_two: CLIPTokenizer associato a text_encoder_two.
        placeholder_token: Token speciale, es. "<cat2>".
        load_path: Percorso del file .safetensors da caricare.
        device: Dispositivo di destinazione ("cpu" o "cuda").
    """
    tensors = load_file(load_path)

    id1 = tokenizer_one.convert_tokens_to_ids(placeholder_token)
    id2 = tokenizer_two.convert_tokens_to_ids(placeholder_token)

    with torch.no_grad():
        text_encoder_one.get_input_embeddings().weight[id1] = tensors["clip_l"].to(
            dtype=text_encoder_one.dtype, device=device
        )
        text_encoder_two.get_input_embeddings().weight[id2] = tensors["clip_g"].to(
            dtype=text_encoder_two.dtype, device=device
        )

    print(f"Embedding caricato da {load_path}")
