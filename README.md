## Folder Structure
```text
└── config.yaml
├── evaluation/
│   ├── metrics/
│   │   ├── editability/
│   │   ├── fidelity/
│   │   │   └── evaluate_fidelity.ipynb
│   │   │   └── gen_fidelity_imgs.ipynb
│   │   │   ├── images_lora/
│   │   │   ├── images_base/
│   │   ├── preservability/
│   │   │   └── evaluate_preservability.ipynb
│   │   │   └── gen_preservability_imgs.ipynb
│   │   │   ├── images_lora/
│   │   │   │   ├── cats/
│   │   │   ├── images_base/
│   │   │   │   ├── cats/
├── data/
│   ├── class_images/
│   ├── instance_images/
│   ├── random_images/
├── models/
│   ├── pretrained/
│   │   ├── sdxl-base/
│   ├── trained/
│   │   ├── lora_cat_v1/
├── experiments/
│   └── gen_random_imgs.ipynb
│   ├── lora/
│   │   ├── train/
│   │   │   └── launcher.py
│   │   │   └── train_lora.sh
│   │   │   └── train_lora_script.py
│   │   ├── test/
│   │   │   └── test_lora.ipynb
│   │   │   ├── test_images/
```
