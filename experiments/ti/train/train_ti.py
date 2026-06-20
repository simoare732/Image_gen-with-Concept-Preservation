"""
Training script per Textual Inversion su Stable Diffusion XL (SDXL).

Utilizzo consigliato tramite launcher:
    python experiments/ti/train/launcher.py

Oppure direttamente:
    accelerate launch experiments/ti/train/train_ti.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --train_data_dir data/instance_images \
        --placeholder_token "<cat2>" \
        --initializer_token "cat" \
        --output_dir models/trained/ti_cat_v1 \
        --max_train_steps 5000 \
        --learning_rate 5e-4 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --mixed_precision fp16 \
        --resolution 1024 \
        --checkpointing_steps 500 \
        --validation_prompt "a photo of <cat2> on the moon" \
        --validation_steps 500 \
        --use_8bit_adam \
        --gradient_checkpointing

Reference ufficiale:
    https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/train_textual_inversion_sdxl.py
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from utils.dataset import TextualInversionDataset
from utils.embedding_handler import save_learned_embeddings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argomenti da riga di comando
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Textual Inversion su SDXL")

    parser.add_argument("--model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, default="<cat2>")
    parser.add_argument("--initializer_token", type=str, default="cat")
    parser.add_argument("--learnable_property", type=str, default="object",
                        choices=["object", "style"])
    parser.add_argument("--output_dir", type=str, default="models/trained/ti_cat_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataset_repeats", type=int, default=100)
    parser.add_argument("--num_vectors", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    # Validation
    parser.add_argument("--validation_prompt", type=str, default=None,
                        help="Prompt usato per generare immagini di validazione durante il training.")
    parser.add_argument("--validation_steps", type=int, default=500,
                        help="Genera immagini di validazione ogni N step.")
    parser.add_argument("--num_validation_images", type=int, default=4)
    # Ottimizzazione memoria
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Usa AdamW a 8-bit (bitsandbytes) per ridurre l'uso di VRAM.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Abilita gradient checkpointing per ridurre l'uso di VRAM.")
    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'latest' oppure path diretto a learned_embeds_stepN.safetensors.")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Step di partenza (auto-rilevato dal nome file se 'latest').")
    # Logging
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb", "none"])

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_text_embeddings(input_ids_one, input_ids_two, text_encoder_one, text_encoder_two):
    """
    Calcola gli embedding testuali concatenati per SDXL (dual-encoder).

    CLIP-L  → penultimo hidden state → [B, 77, 768]
    OpenCLIP-G → penultimo hidden state → [B, 77, 1280] + pooled → [B, 1280]
    Concatenati → prompt_embeds [B, 77, 2048]
    """
    out1 = text_encoder_one(input_ids_one, output_hidden_states=True)
    prompt_embeds = out1.hidden_states[-2]

    out2 = text_encoder_two(input_ids_two, output_hidden_states=True)
    pooled_prompt_embeds = out2[0]
    prompt_embeds_2 = out2.hidden_states[-2]

    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype, device):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    return torch.tensor([add_time_ids], dtype=dtype, device=device)


def log_validation(args, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two,
                   unet, vae, accelerator, global_step):
    """Genera immagini di validazione e le logga. Stessa struttura di log_validation."""
    logger.info(
        f"Running validation (step {global_step})... "
        f"Generating {args.num_validation_images} images with prompt: '{args.validation_prompt}'"
    )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder_one),
        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        vae=vae,
        torch_dtype=torch.float16,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = [
        pipeline(
            args.validation_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        for _ in range(args.num_validation_images)
    ]

    # Salva su disco
    val_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(val_dir, f"step{global_step}_{i}.png"))

    # Logga su tracker (tensorboard / wandb)
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            import numpy as np
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb
            tracker.log({
                "validation": [
                    wandb.Image(img, caption=f"{i}: {args.validation_prompt}")
                    for i, img in enumerate(images)
                ]
            })

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Accelerate setup ---
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    log_with = args.report_to if args.report_to != "none" else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion_sdxl")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Caricamento componenti SDXL ---
    logger.info(f"Caricamento modello: {args.model_id}")
    tokenizer_one = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", variant="fp16"
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.model_id, subfolder="text_encoder_2", variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", variant="fp16")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", variant="fp16")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # --- Aggiunta placeholder token ---
    placeholder_tokens = [args.placeholder_token]
    for i in range(1, args.num_vectors):
        placeholder_tokens.append(f"{args.placeholder_token}_{i}")

    tokenizer_one.add_tokens(placeholder_tokens)
    tokenizer_two.add_tokens(placeholder_tokens)
    placeholder_token_ids_one = tokenizer_one.convert_tokens_to_ids(placeholder_tokens)
    placeholder_token_ids_two = tokenizer_two.convert_tokens_to_ids(placeholder_tokens)

    text_encoder_one.resize_token_embeddings(len(tokenizer_one))
    text_encoder_two.resize_token_embeddings(len(tokenizer_two))

    # --- Inizializzazione embedding ---
    init_id_one = tokenizer_one.convert_tokens_to_ids(args.initializer_token)
    init_id_two = tokenizer_two.convert_tokens_to_ids(args.initializer_token)
    with torch.no_grad():
        init_one = text_encoder_one.get_input_embeddings().weight[init_id_one].clone()
        for ph_id in placeholder_token_ids_one:
            text_encoder_one.get_input_embeddings().weight[ph_id] = init_one.clone()
        init_two = text_encoder_two.get_input_embeddings().weight[init_id_two].clone()
        for ph_id in placeholder_token_ids_two:
            text_encoder_two.get_input_embeddings().weight[ph_id] = init_two.clone()

    logger.info(f"Token '{args.placeholder_token}' inizializzato da '{args.initializer_token}'")

    # --- Freezing ---
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_one.get_input_embeddings().requires_grad_(True)
    text_encoder_two.get_input_embeddings().requires_grad_(True)
    # I parametri addestrabili devono essere in fp32: il GradScaler non supporta gradienti fp16
    text_encoder_one.get_input_embeddings().to(torch.float32)
    text_encoder_two.get_input_embeddings().to(torch.float32)

    # --- Resume from checkpoint ---
    if args.resume_from_checkpoint is not None:
        import glob
        import re
        from safetensors.torch import load_file as _load_sf

        ckpt = args.resume_from_checkpoint
        if ckpt == "latest":
            files = glob.glob(os.path.join(args.output_dir, "learned_embeds_step*.safetensors"))
            if files:
                def _step(p):
                    m = re.search(r'step(\d+)', p)
                    return int(m.group(1)) if m else 0
                ckpt = max(files, key=_step)
                args.start_step = _step(ckpt)
            else:
                logger.warning("Nessun checkpoint trovato in %s, training da zero.", args.output_dir)
                ckpt = None
        elif args.start_step == 0:
            m = re.search(r'step(\d+)', ckpt)
            if m:
                args.start_step = int(m.group(1))

        if ckpt:
            logger.info("Caricamento checkpoint: %s (step %d)", ckpt, args.start_step)
            tensors = _load_sf(ckpt)
            with torch.no_grad():
                for ph_id in placeholder_token_ids_one:
                    text_encoder_one.get_input_embeddings().weight[ph_id] = tensors["clip_l"].to(torch.float32)
                for ph_id in placeholder_token_ids_two:
                    text_encoder_two.get_input_embeddings().weight[ph_id] = tensors["clip_g"].to(torch.float32)

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder_one.gradient_checkpointing_enable()
        text_encoder_two.gradient_checkpointing_enable()

    # --- Ottimizzatore ---
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Usando AdamW 8-bit (bitsandbytes)")
        except ImportError:
            logger.warning("bitsandbytes non trovato, fallback su AdamW standard.")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        list(text_encoder_one.get_input_embeddings().parameters()) +
        list(text_encoder_two.get_input_embeddings().parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # --- Dataset e DataLoader ---
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        placeholder_token=args.placeholder_token,
        learnable_property=args.learnable_property,
        size=args.resolution,
        repeats=args.dataset_repeats,
        center_crop=args.center_crop,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # --- LR Scheduler ---
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --- Prepare con Accelerate ---
    text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    )
    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device)

    # --- Training loop ---
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Inizio training *****")
    logger.info(f"  Immagini di riferimento: {train_dataset.num_images}")
    logger.info(f"  Dataset length (con repeats): {len(train_dataset)}")
    logger.info(f"  Batch size effettivo: {total_batch_size}")
    logger.info(f"  Step totali: {args.max_train_steps}")
    logger.info(f"  LR: {args.learning_rate}")
    logger.info(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    logger.info(f"  8-bit Adam: {args.use_8bit_adam}")

    global_step = args.start_step
    progress_bar = tqdm(
        range(args.start_step, args.max_train_steps),
        desc="Step",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(math.ceil(args.max_train_steps / num_update_steps_per_epoch)):
        text_encoder_one.train()
        text_encoder_two.train()

        for batch in train_dataloader:
            with accelerator.accumulate(text_encoder_one, text_encoder_two):

                # 1. Encode immagini in latent space
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 2. Rumore e timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Conditioning SDXL
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    batch["input_ids_one"].to(accelerator.device),
                    batch["input_ids_two"].to(accelerator.device),
                    text_encoder_one,
                    text_encoder_two,
                )
                add_time_ids = get_add_time_ids(
                    original_size=(args.resolution, args.resolution),
                    crops_coords_top_left=(0, 0),
                    target_size=(args.resolution, args.resolution),
                    dtype=prompt_embeds.dtype,
                    device=accelerator.device,
                ).repeat(bsz, 1)

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # 4. Forward UNet (frozen — requires_grad_(False) impedisce aggiornamento pesi,
                #    ma il gradiente deve fluire attraverso l'UNet fino agli embedding TI)
                model_pred = unet(
                    noisy_latents.to(unet.dtype),
                    timesteps,
                    encoder_hidden_states=prompt_embeds.to(unet.dtype),
                    added_cond_kwargs={k: v.to(unet.dtype) for k, v in added_cond_kwargs.items()},
                ).sample

                # 5. Loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"prediction_type sconosciuto: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                # 6. Zero out gradienti per token non-placeholder (fondamentale in TI)
                if accelerator.sync_gradients:
                    emb_one = accelerator.unwrap_model(text_encoder_one).get_input_embeddings()
                    if emb_one.weight.grad is not None:
                        mask = torch.ones(len(tokenizer_one), dtype=torch.bool, device=emb_one.weight.device)
                        for ph_id in placeholder_token_ids_one:
                            mask[ph_id] = False
                        emb_one.weight.grad[mask] = 0.0

                    emb_two = accelerator.unwrap_model(text_encoder_two).get_input_embeddings()
                    if emb_two.weight.grad is not None:
                        mask = torch.ones(len(tokenizer_two), dtype=torch.bool, device=emb_two.weight.device)
                        for ph_id in placeholder_token_ids_two:
                            mask[ph_id] = False
                        emb_two.weight.grad[mask] = 0.0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.detach().item()})
                accelerator.log({"loss": loss.detach().item()}, step=global_step)

                # Checkpoint intermedio
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    token_name = args.placeholder_token.strip("<>").replace(" ", "_")
                    ckpt_path = os.path.join(args.output_dir, f"learned_embeds_step{global_step}.safetensors")
                    save_learned_embeddings(
                        accelerator.unwrap_model(text_encoder_one),
                        accelerator.unwrap_model(text_encoder_two),
                        tokenizer_one, tokenizer_two,
                        args.placeholder_token, ckpt_path,
                    )

                # Validation
                if (args.validation_prompt and
                        args.validation_steps > 0 and
                        global_step % args.validation_steps == 0 and
                        accelerator.is_main_process):
                    log_validation(
                        args,
                        text_encoder_one, text_encoder_two,
                        tokenizer_one, tokenizer_two,
                        unet, vae, accelerator, global_step,
                    )
                    text_encoder_one.train()
                    text_encoder_two.train()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # --- Salvataggio finale ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "learned_embeds_final.safetensors")
        save_learned_embeddings(
            accelerator.unwrap_model(text_encoder_one),
            accelerator.unwrap_model(text_encoder_two),
            tokenizer_one, tokenizer_two,
            args.placeholder_token, final_path,
        )
        logger.info(f"Training completato! Embedding finale: {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
