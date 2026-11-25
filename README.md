# LiteSSMPolypNet: Code and Experimental Materials

**Manuscript:** LiteSSMPolypNet: A Controlled Evaluation of Training Augmentation and Test-Time Augmentation for Cross-Dataset Polyp Segmentation with Boundary-Aware Analysis  
**Journal:** Visual Computing for Industry, Biomedicine, and Art  
**Manuscript ID:** VICO-D-26-00351R1  
**Authors:** Chiagoziem C. Ukwuoma; Seth Larweh Kodjiku; Stacy E. B. Aggrey; Gyarteng E. S. Addai; Joel V. Dossa; Dongsheng Cai

## Repository contents

- `code/Training_Full_Model.ipynb` — supplied end-to-end notebook for data loading, model definition, training, evaluation, TTA, per-image metrics, and efficiency measurement.
- `analysis/` — supplied scripts for boundary-metric summaries/plots, efficiency summaries/plots, slopegraph visualization, and Wilcoxon/effect-size analyses.
- `docs/` — notes on the data layout, reported settings, and limitations of the supplied materials.
- `requirements.txt` — package names imported by the supplied code; versions were not recorded in the supplied files.

## Dataset arrangement expected by the notebook

The notebook points to a local dataset root named `Poly_New`. Update the `POLYGAP_ROOT` variable in the notebook to the location of your own copy. The loader expects the following structure:

```text
Poly_New/
├── train_set/
│   ├── images/
│   └── masks/
└── test_set/
    ├── Kvasir/
    │   ├── images/
    │   └── masks/
    ├── CVC-ClinicDB/
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/
    │   ├── images/
    │   └── masks/
    ├── ETIS-LaribPolypDB/
    │   ├── images/
    │   └── masks/
    └── CVC-300/
        ├── images/
        └── masks/
```

The notebook's loader reads the existing `train_set` and `test_set` directories. It creates a validation subset from `train_set` using an 80:20 split (`val_split=0.2`) and `random_state=42`; the test directories are loaded separately and are not included in that split. Keep the dataset-owner-provided partitions intact when arranging these folders. The dataset itself is not included in this repository package.

## Running the notebook

1. Create a Python environment and install the packages listed in `requirements.txt`.
2. Install a PyTorch build compatible with your CUDA/CPU environment, following the official PyTorch installation instructions.
3. Place the dataset in the directory structure above.
4. Open `code/Training_Full_Model.ipynb`.
5. Change `POLYGAP_ROOT` to your dataset root.
6. Run the notebook cells in order.

The supplied notebook uses image size 512, batch size 6, a maximum of 500 epochs, AdamW, learning rate `1e-5` in the training call, weight decay `1e-4`, BCE–Dice loss with equal weights, threshold 0.5, and early-stopping patience 100. The loader uses a 20% validation split from `train_set` with seed 42.

## Test-time augmentation

The notebook's TTA implementation evaluates six views: identity, horizontal flip, vertical flip, 90-degree rotation, 180-degree rotation, and 270-degree rotation. It inverse-transforms the predictions to the original orientation and averages the logits before applying the sigmoid and threshold.

## Efficiency measurement implementation

The notebook's efficiency function records parameter count, standard-inference latency, TTA latency, and peak GPU memory. It uses 10 warm-up iterations, up to 30 timed batches, CUDA events, and CUDA synchronization. The supplied efficiency CSVs are included in `results/`.

## Analysis scripts

The scripts in `analysis/` read CSV files using paths specified near the top of each script. Update those paths to match the location of the relevant files in `results/` or your generated per-image result folders before running them. 

## Important reproducibility notes

- The uploaded notebook contains a local Windows dataset path that must be changed by each user.
- The notebook contains selectable architecture configurations. The currently selected configuration in the supplied notebook is `A3_Attn` (with `use_attn=False`); users must verify the selected configuration against the experiment they intend to reproduce before running training. Do not assume that the notebook's currently selected configuration automatically corresponds to every manuscript experiment.
- The supplied notebook includes code to save model weights and result CSV files, but no trained model checkpoint was supplied in this package.

## Citation

If you use this code or experimental materials, please cite the manuscript:

Ukwuoma, C. C., Kodjiku, S. L., Aggrey, S. E. B., Addai, G. E. S., Dossa, J. V., & Cai, D. *LiteSSMPolypNet: A Controlled Evaluation of Training Augmentation and Test-Time Augmentation for Cross-Dataset Polyp Segmentation with Boundary-Aware Analysis*. Visual Computing for Industry, Biomedicine, and Art. 
