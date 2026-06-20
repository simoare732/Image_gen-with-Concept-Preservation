"""
Script di inferenza per Textual Inversion su SDXL.
Legge i path da config.yaml (root del progetto).

Utilizzo:
    python experiments/ti/test/inference_ti.py
    python experiments/ti/test/inference_ti.py --prompts "a photo of <cat2> on a beach"
    python experiments/ti/test/inference_ti.py --use_default_prompts
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

# Prompt di default per la demo
DEFAULT_PROMPTS = [
    "A photo of {} on the moon",
    "A Renaissance oil painting of {}",
    "A photo of {} on a street",
    "A photo of {} with a man",
    "App icon of {}",
    "A {} themed lunchbox",
    "A photo of a golden retriever dog",       # preservazione: altri animali
    "A beautiful mountain landscape at sunset", # preservazione: leakage check
    "A photo of a cat",                         # prior: gatti generici
    "A photo of a beautiful cat",
]

CONFIG_PATH = "/work/cvcs2026/stochastic_parrots/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Inferenza Textual Inversion SDXL")
    parser.add_argument("--prompts", type=str, nargs="+",
                        help="Prompt personalizzati. Usa il placeholder token nel testo.")
    parser.add_argument("--use_default_prompts", action="store_true",
                        help="Usa il set di prompt predefiniti per demo.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pipeline_with_embedding(model_path, embedding_path, placeholder_token, device, dtype):
    """Carica la pipeline SDXL e inietta gli embedding appresi."""
    print(f"Caricamento pipeline da {model_path}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    ).to(device)

    pipe.tokenizer.add_tokens([placeholder_token])
    pipe.tokenizer_2.add_tokens([placeholder_token])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))

    token_id_one = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)
    token_id_two = pipe.tokenizer_2.convert_tokens_to_ids(placeholder_token)

    print(f"Caricamento embedding da {embedding_path}...")
    tensors = load_file(embedding_path)

    with torch.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id_one] = (
            tensors["clip_l"].to(dtype=pipe.text_encoder.dtype, device=device)
        )
        pipe.text_encoder_2.get_input_embeddings().weight[token_id_two] = (
            tensors["clip_g"].to(dtype=pipe.text_encoder_2.dtype, device=device)
        )

    print(f"Token '{placeholder_token}' iniettato correttamente.")
    return pipe


def main():
    args = parse_args()
    cfg = load_config()

    model_path = cfg["paths"]["base_model_dir"]
    embedding_path = os.path.join(
        cfg["paths"]["tiv1_model_dir"],
        cfg["weights_file"]["ti_weights"]
    )
    output_dir = cfg["paths"]["ti_output_test_dir"]
    placeholder_token = cfg["hyperparameters_ti"]["placeholder_token"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Costruisci la lista dei prompt
    if args.use_default_prompts:
        prompts = [p.format(placeholder_token) for p in DEFAULT_PROMPTS]
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = [p.format(placeholder_token) for p in DEFAULT_PROMPTS]

    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline_with_embedding(model_path, embedding_path, placeholder_token, device, dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for i, prompt in enumerate(prompts):
        print(f"Generating image for prompt {i+1}/{len(prompts)}: '{prompt}'")
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            generator=generator,
        ).images[0]
        filename = os.path.join(output_dir, f"prompt_{i+1}.png")
        image.save(filename)

    print(f"\nDone! {len(prompts)} immagini salvate in '{output_dir}'")


if __name__ == "__main__":
    main()
