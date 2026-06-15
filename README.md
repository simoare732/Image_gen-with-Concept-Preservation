# Image Generation with Concept Preservation

A comparative study of personalized image generation using **Textual Inversion (TI)** and **DreamBooth + LoRA** on Stable Diffusion XL (SDXL). The project evaluates both methods on a custom cat concept using four metrics: CLIP-I, CLIP-T, FID, and LPIPS.

---

## Methods

| Method | Trainable Parameters | Steps | Personalization Token |
|--------|---------------------|-------|-----------------------|
| DreamBooth + LoRA | LoRA adapters in UNet (~millions) | ~400 | `TOK cat` |
| Textual Inversion | 2 embedding vectors (768d + 1280d) | 5000 | `<cat2>` |

---

## Folder Structure

```text
├── README.md
├── config.yaml                          # LoRA training config (shared)
├── config_ti_personal.yaml              # TI training config (personal for TI)
├── requirements.txt
│
├── data/
│   ├── download_dataset.py              # Downloads instance images from HuggingFace
│   ├── instance_images/                 # Reference photos of the target concept
│   ├── class_images/                    # Prior preservation images (LoRA)
│   └── random_images/
│
├── experiments/
│   ├── gen_random_imgs.ipynb
│   ├── lora/
│   │   ├── train/
│   │   │   ├── launcher.py
│   │   │   ├── train_lora.sh
│   │   │   └── train_dreambooth_lora_sdxl.py
│   │   └── test/
│   │       ├── test_lora.ipynb
│   │       └── test_images/
│   └── ti/
│       ├── train/
│       │   ├── launcher.py              # Reads config, launches train_ti.py via accelerate
│       │   ├── train_ti.sh              # SLURM job script
│       │   ├── train_ti.py              # Textual Inversion training (SDXL dual-encoder)
│       │   └── utils/
│       │       ├── dataset.py           # TextualInversionDataset with prompt templates
│       │       └── embedding_handler.py # Save/load embeddings to .safetensors
│       └── test/
│           ├── test_ti.ipynb            # Visual inspection of learned concept
│           ├── inference_ti.py          # Standalone inference script
│           └── test_images/
│
├── evaluation/
│   ├── run_generation.sh                # SLURM array job for LoRA image generation
│   ├── all_generation.sh                # Submits all LoRA generation jobs
│   ├── metrics/
│   │   ├── gen_parallel.py              # LoRA: generates images for each metric
│   │   ├── gen_parallel_ti.py           # TI: generates images for each metric
│   │   ├── run_generation_ti.sh         # SLURM array job for TI image generation
│   │   ├── all_generation_ti.sh         # Submits all TI generation jobs
│   │   ├── evaluation_metric.ipynb      # Computes metrics for LoRA vs SDXL baseline
│   │   ├── evaluation_metric_ti.ipynb   # Computes metrics for TI vs SDXL baseline
│   │   ├── sdxl/
│   │   │   ├── clipi/
│   │   │   ├── clipt/
│   │   │   ├── fid/
│   │   │   └── lpips/
│   │   ├── lora-v1/
│   │   │   ├── clipi/
│   │   │   ├── clipt/
│   │   │   ├── fid/
│   │   │   └── lpips/
│   │   ├── lora-v2/
│   │   │   ├── clipi/
│   │   │   ├── clipt/
│   │   │   ├── fid/
│   │   │   └── lpips/
│   │   └── ti-v1/
│   │       ├── clipi/
│   │       ├── clipt/
│   │       ├── fid/
│   │       └── lpips/
│   └── prompts/
│       ├── clipi/
│       │   ├── sdxl/prompt.txt          # Generic cat prompts
│       │   ├── lora/prompt.txt          # Prompts with TOK cat
│       │   └── ti/prompt.txt            # Prompts with <cat2>
│       ├── clipt/
│       │   ├── prompt.txt               # 25 generic prompts (language drift test)
│       │   └── ti/prompt.txt
│       ├── fid/
│       │   ├── prompt.txt               # "A photo of a cat"
│       │   └── ti/prompt.txt            # "A photo of <cat2>"
│       └── lpips/
│           ├── sdxl/prompt.txt
│           ├── lora/prompt.txt
│           └── ti/prompt.txt
│
└── models/
    ├── pretrained/
    │   └── sdxl-base/                   # SDXL base model weights (not tracked by git)
    └── trained/
        ├── lora_cat_v1/                 # Trained LoRA weights (not tracked by git)
        ├── lora_cat_v2/
        └── ti_cat_v1/                   # Learned TI embeddings — learned_embeds_final.safetensors
```

---

## How to Run

> All commands are meant to be run on the cluster after SSH login.
> Training and generation require a GPU node via SLURM.

### 1. Setup

```bash
# Create virtual environment and install dependencies
python3 -m venv /homes/<username>/cvcs2026/venv
source /homes/<username>/cvcs2026/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Download the dataset
python data/download_dataset.py
```

### 2. Training

| Method | Config file | Command |
|--------|-------------|---------|
| DreamBooth + LoRA | `config.yaml` | `sbatch experiments/lora/train/train_lora.sh` |
| Textual Inversion | `config_ti_personal.yaml` → upload to `/homes/<username>/cvcs2026/config_ti.yaml` | `sbatch experiments/ti/train/train_ti.sh` |

### 3. Visual Test (optional)

Launch a JupyterLab session on a GPU node and open the relevant notebook:

| Method | Notebook |
|--------|----------|
| LoRA | `experiments/lora/test/test_lora.ipynb` |
| TI | `experiments/ti/test/test_ti.ipynb` |

### 4. Generate Evaluation Images

```bash
# LoRA
cd evaluation/metrics
bash all_generation.sh          # submits 4 SLURM array jobs (50 tasks each)

# Textual Inversion
bash all_generation_ti.sh       # submits 4 SLURM array jobs (50 tasks each)
```

Monitor progress: `squeue -u $USER`

### 5. Compute Metrics

Open the evaluation notebooks in JupyterLab and run all cells:

| Method | Notebook | Reads from |
|--------|----------|------------|
| LoRA | `evaluation/metrics/evaluation_metric.ipynb` | `evaluation/metrics/lora-v*/` and `sdxl/` |
| TI | `evaluation/metrics/evaluation_metric_ti.ipynb` | `evaluation/metrics/ti-v1/` and `sdxl/` |

Results are printed at the end of each notebook as a summary table.

---

## Evaluation Metrics

| Metric | Measures | Better |
|--------|----------|--------|
| **CLIP-I** | Subject fidelity — cosine similarity between generated and reference images | ↑ higher |
| **CLIP-T** | Language drift — whether the model still generates non-concept prompts correctly | ↑ higher |
| **FID** | Image quality — distributional distance between generated and reference images | ↓ lower |
| **LPIPS** | Diversity — perceptual distance between images generated from the same prompt | ↑ higher |

---

## Results

| Metric | SDXL Base | LoRA v1 | LoRA v2 | TI v1 |
|--------|-----------|---------|---------|-------|
| CLIP-I (↑) | baseline | +16.4% | +14.5% | +8.31% |
| CLIP-T (↑) | 0.2497 | -1.20% drift | -1.33% drift | 0.00% drift |
| FID (↓) | baseline | 56.28 | 65.54 | 90.59 |
| LPIPS (↑) | 0.5234 | 0.5319 | 0.5457 | 0.5484 |

**Key takeaways:**
- **LoRA** achieves higher concept fidelity (CLIP-I) but introduces minor language drift and higher FID
- **TI** guarantees zero language drift and minimal footprint (few KB file), at the cost of lower concept fidelity
- Both methods preserve generative diversity (LPIPS) comparably to the SDXL baseline
