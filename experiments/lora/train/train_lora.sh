#!/bin/bash
#SBATCH --job-name=lora_cat_v2
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=training_output.log

# Attiva l'ambiente
source /homes/saresta/cvcs2026/venv/bin/activate

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="/work/cvcs2026/stochastic_parrots/models/pretrained/sdxl-base" \
  --variant="fp16" \
  --instance_data_dir="/work/cvcs2026/stochastic_parrots/data/instance_images" \
  --class_data_dir="/work/cvcs2026/stochastic_parrots/data/class_images" \
  --output_dir="/work/cvcs2026/stochastic_parrots/models/trained/lora_cat_v2" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of TOK cat" \
  --class_prompt="a photo of a cat" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --max_train_steps=500 \
  --validation_prompt="A photo of TOK cat on the moon" \
  --validation_epochs=25 \
  --rank=4 \
  --with_prior_preservation --prior_loss_weight=2.0 \
  --checkpointing_steps=250