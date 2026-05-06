## Folder Structure
```text
├── README.md
├── config.yaml
├── data/
│   ├── class_images/
│   ├── instance_images/
│   ├── random_images/
├── evaluation/
│   ├── metrics/
│   │   ├── evaluation_metric.ipynb
│   │   ├── gen_parallel.py
│   │   ├── sdxl
│   │   │   ├── clipi/
│   │   │   ├── clipt/
│   │   │   ├── fid/
│   │   │   ├── lpips/
│   │   ├── lora-v1
│   │   │   ├── clipi/
│   │   │   ├── clipt/
│   │   │   ├── fid/
│   │   │   ├── lpips/
│   │   ├── lora-v2
│   │   │   ├── clipi/
│   │   │   ├── clipt/
│   │   │   ├── fid/
│   │   │   ├── lpips/
│   ├── prompts/
│   │   ├── clipi/
│   │   ├── clipt/
│   │   ├── fid/
│   │   ├── lpips/
├── experiments/
│   ├── gen_random_imgs.ipynb
│   ├── lora/
│   │   ├── test/
│   │   │   ├── test_lora.ipynb
│   │   │   ├── test_images/
│   │   ├── train/
│   │   │   ├── launcher.py
│   │   │   ├── train_lora.sh
│   │   │   └── train_dreambooth_lora_sdxl.py
├── models/
│   ├── pretrained/
│   │   ├── sdxl-base/
│   ├── trained/
│   │   ├── lora_cat_v1/
│   │   ├── lora_cat_v2/