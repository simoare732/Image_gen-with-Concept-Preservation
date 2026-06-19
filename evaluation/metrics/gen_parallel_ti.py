'''
Generates images for evaluation metrics using Textual Inversion (TI) or SDXL base.
Run via SLURM array job using run_generation_ti.sh:
    sbatch run_generation_ti.sh tiv1 fid

Parameters:
- model:  sdxl, tiv1
- metric: fid, clipt, clipi, lpips
'''

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import os
import gc
import yaml
import math
import argparse

with open("/homes/ktotaro/cvcs2026/config_ti.yaml", "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Generation of images for evaluation metrics (TI).")
parser.add_argument("--model", type=str, choices=["sdxl", "tiv1"], required=True)
parser.add_argument("--metric", type=str, choices=["fid", "clipt", "clipi", "lpips"], required=True)
args = parser.parse_args()

MODEL_PATH = config["paths"]["base_model_dir"]
TI_EMBEDDING_PATH = os.path.join(
    config["paths"]["tiv1_model_dir"],
    config["weights_file"]["ti_weights"]
)
PLACEHOLDER_TOKEN = config["hyperparameters"]["placeholder_token"]
EVALUATION_DIR = config["paths"]["evaluation_dir"]
PROMPTS_DIR = config["paths"]["prompts_dir"]

if args.model == "tiv1":
    model_type = "ti"
    output_model_tag = "ti-v1"
else:
    model_type = "sdxl"
    output_model_tag = "sdxl"

OUTPUT_DIR = os.path.join(EVALUATION_DIR, "metrics", output_model_tag, args.metric)

INFERENCE_STEPS = 50

number_images = {
    "fid":   5000,
    "clipt": 100,
    "clipi": 100,
    "lpips": 1000,
}

# ── Model loading ────────────────────────────────────────────────────────────

pipe = None

def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_pipe():
    global pipe
    if pipe is None:
        clean_cache()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
    return pipe


def sdxl_model():
    p = get_pipe()
    clean_cache()
    return p


def ti_model():
    p = get_pipe()
    clean_cache()

    p.tokenizer.add_tokens([PLACEHOLDER_TOKEN])
    p.tokenizer_2.add_tokens([PLACEHOLDER_TOKEN])
    p.text_encoder.resize_token_embeddings(len(p.tokenizer))
    p.text_encoder_2.resize_token_embeddings(len(p.tokenizer_2))

    token_id_one = p.tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    token_id_two = p.tokenizer_2.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

    tensors = load_file(TI_EMBEDDING_PATH)
    with torch.no_grad():
        p.text_encoder.get_input_embeddings().weight[token_id_one] = (
            tensors["clip_l"].to(dtype=p.text_encoder.dtype, device="cuda")
        )
        p.text_encoder_2.get_input_embeddings().weight[token_id_two] = (
            tensors["clip_g"].to(dtype=p.text_encoder_2.dtype, device="cuda")
        )

    return p

# ── Prompt loading ───────────────────────────────────────────────────────────

def load_prompts(metric, model):
    base_prompt_dir = os.path.join(PROMPTS_DIR, metric)
    model_dir = os.path.join(base_prompt_dir, model)

    if os.path.isdir(model_dir):
        prompt_file = os.path.join(model_dir, "prompt.txt")
    else:
        prompt_file = os.path.join(base_prompt_dir, "prompt.txt")

    print(f"Loading prompts from: {prompt_file}")
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


prompts = load_prompts(args.metric, model_type)

# ── Task list ────────────────────────────────────────────────────────────────

all_tasks = []
SEEDS_PER_PROMPT = number_images[args.metric]

for prompt_idx, prompt_text in enumerate(prompts):
    for seed in range(1, SEEDS_PER_PROMPT + 1):
        i = prompt_idx + 1
        filename = f"image_{i:02d}_{seed:04d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        all_tasks.append((prompt_text, seed, filepath))

TOTAL_IMAGES = len(all_tasks)

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
total_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

chunk_size = math.ceil(TOTAL_IMAGES / total_tasks)
start_idx = task_id * chunk_size
end_idx = min(start_idx + chunk_size, TOTAL_IMAGES)
my_tasks = all_tasks[start_idx:end_idx]

# ── Generation ───────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Job {task_id}/{total_tasks - 1} — generating {len(my_tasks)} images "
      f"(model={args.model}, metric={args.metric})")

if len(my_tasks) > 0:
    if args.model == "sdxl":
        pipeline = sdxl_model()
    else:
        pipeline = ti_model()

for prompt_text, seed, filepath in my_tasks:
    filename = os.path.basename(filepath)

    if os.path.exists(filepath):
        print(f"Skip: {filename} already exists.")
        continue

    print(f"Generating: {filename} (seed={seed})")
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipeline(
        prompt=prompt_text,
        num_inference_steps=INFERENCE_STEPS,
        generator=generator,
    ).images[0]
    image.save(filepath)

print(f"Job {task_id} completed!")
