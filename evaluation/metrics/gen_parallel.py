'''
You can run this script in parallel, running the script "run_generation.sh", with the following command:
sbatch run_generation.sh model metric
An example would be
sbatch run_generation.sh sdxl fid

The parameters can be:
- model: sdxl, lorav1, lorav2
- metric: fid, clipt, clipi, lpips
'''



import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import os
import gc
import yaml
import math
import argparse
from safetensors.torch import load_file

with open(os.path.expanduser("/work/cvcs2026/stochastic_parrots/config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Config arguments
parser = argparse.ArgumentParser(description="Generation of images for evaluation metrics.")
parser.add_argument("--model", type=str, choices=["sdxl", "lorav1", "lorav2", "tiv1"], required=True, help="Which model to use")
parser.add_argument("--metric", type=str, choices=["fid", "clipt", "clipi", "lpips"], required=True, help="For which metric to generate images")
args = parser.parse_args()

LORA_VERSION = 1
if args.model == "lorav2":
    LORA_VERSION = 2

if "lora" in args.model:
    model_type = "lora"
elif args.model == "tiv1":
    model_type = "ti"
else:
    model_type = "sdxl"


# Definition PATHS
TI_EMBEDDING_PATH = os.path.join(
    config["paths"]["tiv1_model_dir"],
    config["weights_file"]["ti_weights"]
)

PLACEHOLDER_TOKEN = config["hyperparameters_ti"]["placeholder_token"]

MODEL_PATH = config["paths"]["base_model_dir"]
LORA_PATH = config["paths"]["lorav" + str(LORA_VERSION) + "_model_dir"]
LORA_WEIGHTS_FILE = config["weights_file"]["lora_weights"]

OUTPUT_CLIPI_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/clipi"
OUTPUT_CLIPI_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v" + str(LORA_VERSION) + "/clipi"
OUTPUT_CLIPI_TI = config["paths"]["evaluation_dir"] + "/metrics/ti-v1/clipi"

OUTPUT_CLIPT_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/clipt"
OUTPUT_CLIPT_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v" + str(LORA_VERSION) + "/clipt"
OUTPUT_CLIPT_TI = config["paths"]["evaluation_dir"] + "/metrics/ti-v1/clipt"

OUTPUT_FID_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/fid"
OUTPUT_FID_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v" + str(LORA_VERSION) + "/fid"
OUTPUT_FID_TI = config["paths"]["evaluation_dir"] + "/metrics/ti-v1/fid"

OUTPUT_LPIPS_SDXL = config["paths"]["evaluation_dir"] + "/metrics/sdxl/lpips"
OUTPUT_LPIPS_LORA = config["paths"]["evaluation_dir"] + "/metrics/lora-v" + str(LORA_VERSION) + "/lpips"
OUTPUT_LPIPS_TI = config["paths"]["evaluation_dir"] + "/metrics/ti-v1/lpips"

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
            "lora": OUTPUT_FID_LORA,
            "ti": OUTPUT_FID_TI},
    "clipt": {"sdxl": OUTPUT_CLIPT_SDXL, 
              "lora": OUTPUT_CLIPT_LORA,
              "ti": OUTPUT_CLIPT_TI},
    "clipi": {"sdxl": OUTPUT_CLIPI_SDXL, 
              "lora": OUTPUT_CLIPI_LORA,
              "ti": OUTPUT_CLIPI_TI},
    "lpips": {"sdxl": OUTPUT_LPIPS_SDXL, 
              "lora": OUTPUT_LPIPS_LORA,
              "ti": OUTPUT_LPIPS_TI}
}



# Helper functions
pipe = None
OUTPUT_DIR = path_mapping[args.metric][model_type]
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

# Function to get Textual Inversion model, it adds the placeholder token to the tokenizer and loads the learned embeddings into the text encoder
def ti_model():
    p = getPipe()
    cleanCache()

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


prompts = load_prompts(args.metric, model_type)


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
    if model_type == "sdxl":
        pipe = sdxlModel()
    elif model_type == "lora":
        pipe = loraModel()
    elif model_type == "ti":
        pipe = ti_model()

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