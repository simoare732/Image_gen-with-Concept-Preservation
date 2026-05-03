'''
You can run this script in parallel, running the script "run_generation.sh", with the following command:
sbatch run_generation.sh model metric
An example would be
sbatch run_generation.sh sdxl fid

The parameters can be:
- model: sdxl, lora
- metric: fid, clipt, clipi, lpips
'''



import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import os
import gc
import yaml
import math
import argparse

with open(os.path.expanduser("/work/cvcs2026/stochastic_parrots/config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Config arguments
parser = argparse.ArgumentParser(description="Generation of images for evaluation metrics.")
parser.add_argument("--model", type=str, choices=["sdxl", "lora"], required=True, help="Which model to use")
parser.add_argument("--metric", type=str, choices=["fid", "clipt", "clipi", "lpips"], required=True, help="For which metric to generate images")
args = parser.parse_args()

# Definition PATHS
MODEL_PATH = config["paths"]["base_model_dir"]
LORA_PATH = config["paths"]["lora_model_dir"]
LORA_WEIGHTS_FILE = config["weights_file"]["lora_weights"]

OUTPUT_CLIPI_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/clipi"
OUTPUT_CLIPI_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v1/clipi"

OUTPUT_CLIPT_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/clipt"
OUTPUT_CLIPT_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v1/clipt"

OUTPUT_FID_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/fid"
OUTPUT_FID_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v1/fid"

OUTPUT_LPIPS_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/lpips"
OUTPUT_LPIPS_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v1/lpips"

PROMPTS_DIR = config["paths"]["prompts_dir"]


INFERENCE_STEPS = 50


number_images = {  # Number of images per prompt
    "fid": 5000,  # 1 prompt x 5000 seeds = 5000 images
    "clipt": 100,  # 25 prompt x 100 seeds = 2500 images
    "clipi": 100, # 20 prompts x 100 seeds = 2000 images
    "lpips": 1000  # 3 prompts x 1000 seeds = 3000 images
}


path_mapping = {
    "fid": {"sdxl": OUTPUT_FID_SDXL, 
            "lora": OUTPUT_FID_LORA},
    "clipt": {"sdxl": OUTPUT_CLIPT_SDXL, 
              "lora": OUTPUT_CLIPT_LORA},
    "clipi": {"sdxl": OUTPUT_CLIPI_SDXL, 
              "lora": OUTPUT_CLIPI_LORA},
    "lpips": {"sdxl": OUTPUT_LPIPS_SDXL, 
              "lora": OUTPUT_LPIPS_LORA}
}



# Helper functions
pipe = None
OUTPUT_DIR = path_mapping[args.metric][args.model]
SEEDS_PER_IMAGES = number_images[args.metric]

# Function to clean GPU cache and collect garbage, useful to prevent memory leaks during model loading/unloading
def cleanCache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# Function to load the base model pipeline, cached globally to avoid redundant loading across different functions
def getPipe():
    global pipe
    if pipe is None:
        cleanCache()
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda") 

    return pipe

# Function to get SDXL model
def sdxlModel():
    pipe = getPipe()

    pipe.unload_lora_weights()
    cleanCache()

    return pipe 


# Function to get LoRA model, it loads the LoRA weights on top of the base model pipeline
def loraModel():
    pipe = getPipe()

    pipe.unload_lora_weights()

    pipe.load_lora_weights(
        LORA_PATH,
        weight_name=LORA_WEIGHTS_FILE
    )   

    cleanCache()

    return pipe


# Function to load the correct prompts according to the metric and model.
def load_prompts(metric, model):
    base_prompt_dir = os.path.join(PROMPTS_DIR, metric)
    model_dir = os.path.join(base_prompt_dir, model)

    if os.path.isdir(model_dir):
        prompt_file = os.path.join(model_dir, "prompt.txt")
    
    else:
        prompt_file = os.path.join(base_prompt_dir, "prompt.txt")

    print(f"Loading prompts from: {prompt_file}")

    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    return prompts


prompts = load_prompts(args.metric, args.model)


# Create a list of all the tasks to be executed, each task is a tuple of (prompt_text, seed, output_filepath)
all_tasks = []
for prompt_idx, prompt_text in enumerate(prompts):
    for seed in range(1, SEEDS_PER_IMAGES + 1):
        i = prompt_idx + 1  # To have prompt indices starting from 1 instead of 0

        filename = f"image_{i:02d}_{seed:04d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        all_tasks.append((prompt_text, seed, filepath))

TOTAL_IMAGES = len(all_tasks)


# Environment Variables of SLURM 
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
total_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

# How many images to generate per task
chunk_size = math.ceil(TOTAL_IMAGES / total_tasks)


# Start/end seed for each task
start_seed = (task_id * chunk_size)
end_seed = min(start_seed + chunk_size, TOTAL_IMAGES)


# Get the subset of tasks for this job
my_tasks = all_tasks[start_seed:end_seed]


print(f"Start Job ID: {task_id}/{total_tasks-1}")
print(f"This job will generate {len(my_tasks)} images (Seed from {start_seed} to {end_seed-1})")

# Load the model once per job
if len(my_tasks) > 0:
    if args.model == "sdxl":
        pipe = sdxlModel()
    elif args.model == "lora":
        pipe = loraModel()


# Generation
for prompt_text, seed, filepath in my_tasks:
    filename = os.path.basename(filepath)

    # Skip an image if it already exists, useful for resuming jobs without overwriting existing images
    if os.path.exists(filepath):
        print(f"Skip: {filename} already exists.")
        continue

    print(f"Generating: {filename} (Seed: {seed})")
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Generazione e salvataggio
    image = pipe(prompt_text, num_inference_steps=INFERENCE_STEPS, generator=generator).images[0]
    image.save(filepath)

print(f"Job {task_id} completed!")