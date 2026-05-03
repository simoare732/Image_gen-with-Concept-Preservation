import yaml
import subprocess
import os

# 1. Load configuration
with open(os.path.expanduser("/work/cvcs2026/stochastic_parrots/config.yaml"), "r") as f:
    cfg = yaml.safe_load(f)


# 2. Build command
filename = "train_lora_script.py"
cmd = [
    "accelerate", "launch", filename,
    f"--pretrained_model_name_or_path={cfg['paths']['base_model_dir']}",
    f"--instance_data_dir={cfg['paths']['instance_images_dir']}",
    f"--class_data_dir={cfg['paths']['class_images_dir']}",
    f"--output_dir={cfg['paths']['lora_model_dir']}",
    f"--learning_rate={cfg['hyperparameters']['learning_rate']}",
    f"--max_train_steps={cfg['hyperparameters']['max_train_steps']}",
    "--instance_prompt=a photo of TOK cat",
    "--class_prompt=a photo of a cat",
    "--validation_prompt=A photo of TOK cat on the moon",
    f"--variant={cfg['hyperparameters']['variant']}",
    f"--mixed_precision={cfg['hyperparameters']['mixed_precision']}",
    f"--resolution={cfg['hyperparameters']['resolution']}",
    "--train_batch_size=1",
    "--use_8bit_adam",
    "--gradient_accumulation_steps=4",
    "--gradient_checkpointing",
    "--with_prior_preservation",
    "--prior_loss_weight=1.0"
    "--rank=8",
    f"--checkpointing_steps={cfg['hyperparameters']['checkpointing_steps']}"
]

# 3. Esegui il comando
print(f"🚀 Run training")
subprocess.run(cmd)


