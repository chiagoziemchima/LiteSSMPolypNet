# Implementation details

This document records settings visible in the supplied `Training_Full_Model.ipynb`. 

## Data and transforms
- Input resize: 512 × 512.
- Training augmentation in `make_transforms`: horizontal flip (p=0.5), vertical flip (p=0.3), RandomRotate90 (p=0.5), affine transformation (p=0.5; translate_percent=(0.05, 0.05), scale=0.10, rotate=15), elastic/grid/optical distortion via `OneOf` (outer p=0.5), brightness/contrast or CLAHE via `OneOf` (outer p=0.5), HueSaturationValue (p=0.5), Gaussian or motion blur via `OneOf` (outer p=0.15), GaussNoise (p=0.5), and image compression (p=0.5).
- Normalization: ImageNet mean (0.485, 0.456, 0.406), standard deviation (0.229, 0.224, 0.225).
- Validation/test transforms: resize, normalization (when enabled), and tensor conversion; training-only augmentations are disabled.
- Batch size: 6.
- Validation fraction from training folder: 0.2; random seed: 42.
- DataLoader `num_workers`: 0; `pin_memory`: True.

## Optimization and training
- Optimizer: AdamW.
- Learning rate in the notebook's training call: 1e-5.
- Weight decay: 1e-4.
- Scheduler: CosineAnnealingWarmRestarts with `T_0=10`, `T_mult=2`.
- Loss: BCE–Dice with weights 0.5 and 0.5; smooth=1.0.
- Maximum epochs: 500.
- Early-stopping patience: 100 epochs, monitored using validation Dice.
- Prediction threshold: 0.5.

## TTA
Six views: identity, horizontal flip, vertical flip, 90°, 180°, and 270° rotation. Inverse transformations are applied to logits before averaging.

## Efficiency benchmark
- CUDA is required by the supplied benchmark function.
- Warm-up: 10 iterations.
- Timed batches: up to 30.
- CUDA events and synchronization are used.
- Outputs: parameter count, latency per image, and peak allocated GPU memory, separately for NoTTA and TTA.

## Configuration caution
The notebook defines `A0_Full`, `A1_MSP`, `A2_SSM`, `A3_Attn`, `A4_Backbone_swap`, and `A5_Decoder_width`. In the supplied notebook, `selected_config` is set to `A3_Attn`, whose dictionary entry sets `use_attn=False`. Confirm the intended configuration before running. 
