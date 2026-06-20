"""
Launcher per il training di Textual Inversion su SDXL.
Legge i parametri da config.yaml (root del progetto) e lancia train_ti.py via accelerate.

Utilizzo (dalla root del progetto):
    python experiments/ti/train/launcher.py
"""

import os
import subprocess
import yaml

CONFIG_PATH = "/work/cvcs2026/stochastic_parrots/config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

p = cfg["paths"]
hp = cfg["hyperparameters_ti"]
wf = cfg["weights_file"]

script_path = os.path.join(os.path.dirname(__file__), "train_ti.py")

cmd = [
    "accelerate", "launch",
    "--mixed_precision", hp['mixed_precision'],  # passa a accelerate, non allo script
    script_path,
    f"--model_id={p['base_model_dir']}",
    f"--train_data_dir={p['instance_images_dir']}",
    f"--output_dir={p['tiv1_model_dir']}",
    f"--placeholder_token={hp['placeholder_token']}",
    f"--initializer_token={hp['initializer_token']}",
    f"--mixed_precision={hp['mixed_precision']}",
    f"--resolution={hp['resolution']}",
    f"--learning_rate={hp['learning_rate']}",
    f"--max_train_steps={hp['max_train_steps']}",
    f"--checkpointing_steps={hp['checkpointing_steps']}",
    "--validation_steps=500",
    "--validation_prompt=a photo of <cat2> on the moon",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=4",
    "--use_8bit_adam",
    "--gradient_checkpointing",
    "--resume_from_checkpoint=latest",
]

print("Avvio training Textual Inversion...")
subprocess.run(cmd)
