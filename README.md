# Multi-Task Multimodal Fusion for Pertussis Booster Response Prediction

Companion code for the preprint:

> Sitani D. *Multi-Task Multimodal Fusion Using Tabular Foundation Models
> for Peak and Durability Prediction in Pertussis Booster Vaccine
> Response*. arXiv (2026).
> https://doi.org/10.48550/arXiv.2605.12852

This repository provides **trained model weights, saved test set
predictions, and the scripts needed to reproduce every table and
figure in the paper**. Training code is held back pending journal
publication.

---

## What's in this repository

```
.
├── README.md                          (this file)
├── requirements.txt
├── reproduce_results.py               regenerate all paper tables / figures
├── models/
│   ├── baseline_model.pt              trained model (seed 13)
│   └── baseline_model_state_dict.pt   state dict for the same model
├── data/
│   ├── test_predictions.csv           saved val + test predictions (p_T1, p_T2)
│   └── dissociation_subjects.csv      peak/retention values per subject
└── results/
    ├── presence_task1.csv             subject-modality presence matrix (Task 1)
    ├── presence_task2.csv             subject-modality presence matrix (Task 2)
    ├── bootstrap_ci.json              bootstrap AUROC arrays + 95% CIs (B = 1000)
    ├── permutation_test_baseline.csv  1000 null AUROCs (T1, T2)
    ├── permutation_summary_baseline.json
    ├── modality_loo.csv               leave-one-out per-modality AUROCs
    ├── modality_koo.csv               keep-one-out per-modality AUROCs
    ├── degradation.csv                graceful-degradation curves
    ├── ablation_results.json          architectural ablation (4 configs)
    └── baseline_comparison.json       LR / XGBoost / TabMLP comparison
```



## Installation

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

Tested on Python 3.10. Reproducing the figures requires only CPU.

---

## Reproducing the paper's tables and figures

```bash
python reproduce_results.py --output-dir figures/
```

This regenerates:
- Fig 1 (cohort × modality missingness)
- Fig 3 (peak / durability dissociation)
- Fig 4 (test ROC with bootstrap CI band)
- Fig 5 (bootstrap AUROC distributions)
- Fig 6 (permutation null distributions)
- Fig 7 (per-modality contribution: LOO + KOO)
- Fig 8 (graceful degradation)
- Table 3 (architectural ablation; LaTeX rows in `table3_ablation.tex`)
- Table 4 (baseline comparison; LaTeX rows in `table4_baselines.tex`)

All from saved JSON / CSV artifacts in `results/` and `data/`. No
retraining is required.

---

## Data

This work uses the public CMI-PB dataset:

> Shinde P, Soldevila F, Reyna J, et al. *A multi-omics systems
> vaccinology resource to develop and test computational models of
> immunity.* Cell Reports Methods (2024).
> doi:10.1016/j.crmeth.2024.100731

The raw and harmonised dataset is
available from [https://www.cmi-pb.org/](https://www.cmi-pb.org/).

---

## Citation

If you use any artifact from this repository, please cite:

```bibtex
@article{sitani2026cmipb,
  title   = {Multi-Task Multimodal Fusion Using Tabular Foundation
             Models for Peak and Durability Prediction in Pertussis
             Booster Vaccine Response},
  author  = {Sitani, Divya},
  journal = {arXiv preprint arXiv:2605.12852},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.12852}
}
```

---

## Contact

Divya Sitani — jrpcsitani@gmail.com

For early access to the training pipeline, please email; collaborations
welcome.
