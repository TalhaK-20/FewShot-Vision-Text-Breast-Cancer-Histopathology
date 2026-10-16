# Toward Reliable Few-Shot Vision–Text Fusion for Breast Cancer Histopathology Classification

Official implementation of the paper *"Toward Reliable Few-Shot Vision–Text Fusion for Breast Cancer Histopathology Classification."*

**Authors:** Talha Khalid, Ashfaq Ahmad, Iqra Javed, Waseem Iqbal, Muhammad Awais Ali, Muhammad Shoaib
University of Management and Technology (UMT), Lahore, Pakistan

---

## Overview

Few-shot classification of breast cancer histopathology images is attractive because annotation is expensive and several tumor subtypes have very few labeled examples. Pani et al. (CBMS 2025) proposed fusing Vision Transformer (ViT) image embeddings with Vision-Language Model (VLM) text descriptions for prototype-based few-shot classification of breast histopathology images.

This repository contains a **critical review** of that framework's publicly released implementation and a **corrected framework** evaluated under a strictly patient-disjoint protocol. The work has four parts:

1. **Audit.** Seven concrete loopholes in the released implementation are identified, each cited against the claim or code path it contradicts.
2. **Corrected framework.** The corrected pipeline combines:
   - a ViT-Small vision branch with genuine pre-logits embeddings
   - an image-conditioned text branch built from CLIP zero-shot retrieval over a histopathology descriptor bank
   - the original Fisher-criterion / canonical-correlation fusion design, applied to the corrected features
3. **Leakage-free protocol.** A patient-level BreakHis split, with an automated check that asserts zero patient overlap.
4. **Honest evaluation.** Multi-episode, multi-magnification results with modality ablations, majority-class baselines and uncertainty estimates.

## Loopholes identified in the base framework

| # | Loophole | Consequence |
|---|---|---|
| A | The text-generation routine computed CLIP image features, then discarded them and assigned the fixed string "Generated text description" to every image | The text modality carried zero information, and the Fisher relevance weighting collapsed to noise |
| B | The paper describes BLIP-2 captioning, but the executed code calls CLIP's `get_image_features` | The described captioning procedure could not have produced the reported descriptions |
| C | The ViT head was replaced with a 2-unit layer and features were read after the head | Prototypes were 2-D classifier logits, not 384-D embeddings |
| D | Support and query images were drawn at image level from a ready-made Train/Test folder | The same patient's tissue appeared on both sides, inflating accuracy |
| E | No random seed and no multi-episode averaging | A single unrepeatable support-set draw was reported |
| F | The full ≈22M-parameter backbone was fine-tuned on 10 images with no augmentation or weight decay | The support set was memorized almost immediately |
| G | Accuracy only, no majority-class baseline, a single magnification, hard-coded paths, no checkpoints | Results could not be properly assessed or reproduced |

## Method

The task is episodic, 2-way (benign / malignant), K-shot classification with K = 5.

- **Vision encoder.** ViT-Small/16 (timm `vit_small_patch16_224`, ImageNet-1k).
  - All parameters are frozen except the final transformer block, the final norm layer and an auxiliary linear head.
  - Embeddings are taken from `forward_head(..., pre_logits=True)`, giving 384 dimensions.
- **Text encoder.** A fixed bank of 16 histopathology morphology phrases is embedded once by a frozen CLIP ViT-B/32.
  - The phrases deliberately never name a class, so no label leaks through this step.
  - The top-k phrases by cosine similarity form each image's description.
  - That description is encoded with SentenceTransformer `all-MiniLM-L6-v2` into 384 dimensions.
- **Fusion.**
  - Each dimension's relevance is scored with Fisher's criterion over the support set.
  - The two modalities' similarity is scored canonical-correlation style.
  - Together these give per-modality weights, and the fused representation is a 768-dimensional vector.
- **Classification.** Nearest class prototype by cosine similarity.

### Patient-level BreakHis split

Patient identifiers are recovered from the official BreakHis filenames. Patients are stratified by class and assigned 70/30 to Train/Test with seed 42, and the same split is reused across all magnifications. Within an episode:
- the K support shots come from K distinct patients;
- every query image from a support patient is excluded.

| Magnification | Train img. | Train pat. | Test img. | Test pat. |
|---|---:|---:|---:|---:|
| 40X | 1,446 | 58 | 549 | 24 |
| 100X | 1,520 | 58 | 561 | 24 |
| 200X | 1,439 | 58 | 574 | 24 |
| 400X | 1,315 | 58 | 505 | 24 |
| All combined | 5,720 | 58 | 2,189 | 24 |

## Results

5-shot results on pooled query and test images. ViT-only accuracy and balanced accuracy are mean ± std over 5 episodes; fusion is a single episode.

| Magnification | Model | Accuracy (%) | Bal. Acc. (%) | F1-macro (%) | ROC-AUC | MCC | Majority BL (%) |
|---|---|---|---|---|---|---|---|
| 40X | ViT-S only | 72.36 ± 4.06 | 71.20 ± 3.39 | 68.38 | 0.796 | 0.396 | 72.66 |
| 40X | Fusion | 81.37 | 70.89 | 73.16 | 0.859 | 0.489 | 72.82 |
| 100X | ViT-S only | 62.56 ± 10.23 | 64.54 ± 6.16 | 59.33 | 0.708 | 0.266 | 74.00 |
| 100X | Fusion | 57.71 | 66.33 | 57.16 | 0.757 | 0.299 | 72.78 |
| 200X | ViT-S only | 68.97 ± 5.49 | 68.01 ± 7.45 | 65.04 | 0.741 | 0.329 | 72.47 |
| 200X | Fusion | 63.84 | 56.82 | 56.55 | 0.624 | 0.132 | 71.98 |
| 400X | ViT-S only | 65.50 ± 7.17 | 67.99 ± 3.44 | 63.20 | 0.755 | 0.333 | 71.94 |
| 400X | Fusion | 73.88 | 66.46 | 67.05 | 0.766 | 0.344 | 70.90 |
| All combined | ViT-S only | 67.22 ± 4.33 | 62.00 ± 5.04 | 60.75 | 0.685 | 0.231 | 72.66 |
| All combined | Fusion | 66.73 | 66.52 | 63.04 | 0.738 | 0.296 | 73.42 |

### Same-episode modality ablation (accuracy, %)

| Magnification | Vision only | Text only | Fused |
|---|---:|---:|---:|
| 40X | 81.72 | 72.23 | 81.37 |
| 100X | 64.13 | 41.78 | 57.71 |
| 200X | 60.99 | 46.64 | **63.84** |
| 400X | 71.21 | 59.83 | **73.88** |
| All combined | 67.42 | 41.51 | 66.73 |

The text branch is genuinely informative but remains the weakest modality, because it is a fixed descriptor bank rather than a learned encoder. Under the same support-set draw, fusion improves over vision-only at 200X (+2.85 points) and 400X (+2.67 points), but not at 40X, 100X or all magnifications combined.

The base paper's reported figures are **not** used as a benchmark. Loopholes A–C mean those figures do not isolate a genuine text-modality contribution.

## Requirements

- Python 3.12
- PyTorch with CUDA (developed with PyTorch 2.11 and CUDA 12.8) and torchvision
- timm 1.0.30
- Hugging Face Transformers 5.17.0 (for CLIP)
- Sentence-Transformers 6.1.0
- scikit-learn 1.9.1, NumPy, pandas, matplotlib, tqdm

```bash
pip install torch torchvision timm transformers sentence-transformers scikit-learn numpy pandas matplotlib tqdm
```

Experiments were run on an AMD Ryzen 7 5800X CPU, 60 GB RAM and an NVIDIA GeForce RTX 3080 (10 GB) under Ubuntu Linux. End to end, the ViT-only pipeline takes about 7.7 minutes and the fusion pipeline about 2.2 minutes.

## Usage

1. Download **BreakHis** from the official source: <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>. Use its binary benign/malignant folders (`classificacao_binaria`).
2. Place the raw data at `./dataset/BreakHis/classificacao_binaria` or `./dataset/classificacao_binaria`.
3. Run either notebook:
   - `Toward_Reliable_Few-Shot_Vision-Text_Fusion_for_Breast_Cancer_Histopathology_Classification.ipynb` runs the **ViT + text fusion** model.
   - `Toward_Reliable_Few-Shot_Vision-Text_Fusion_for_Breast_Cancer_Histopathology_Classification_ViT_Baseline.ipynb` runs the **ViT-only** baseline.

On first run, each notebook builds the patient-level split under `./data/BreakHist_patient_split`, with a `split_manifest.json` for each magnification. It then evaluates all five magnification settings: 40X, 100X, 200X, 400X and all combined.

For each setting, the notebooks save:
- per-episode metrics (JSON and CSV)
- summary tables
- modality ablations (fusion model)
- diagnostic plots: confusion matrix, ROC and precision–recall curves, fine-tuning curve and t-SNE

### Main hyperparameters

| Setting | Value |
|---|---|
| Vision backbone | ViT-Small/16 (timm), ImageNet-1k |
| Fine-tuned parameters | Last transformer block + head only |
| Optimizer / LR / weight decay | Adam / 1×10⁻³ / 1×10⁻⁴ |
| Epochs per episode | 50 |
| Shots per class | 5 |
| Episodes | ViT-only: 5 (seeds 42–46); fusion: 1 (seed 42) |
| Split seed / test fraction | 42 / 0.3 |

The notebooks can run more fusion episodes by setting `N_EPISODES` above 1.

## Limitations

- Fusion is evaluated on a single episode, because CLIP and SentenceTransformer feature extraction dominate its runtime.
- The text branch is a fixed, non-learned descriptor bank, which limits what the text side can contribute.
- The fusion weights are computed analytically and cannot learn to down-weight an uninformative modality.
- The scope is binary classification on a single dataset.

## Citation

If you use this code, please cite the accompanying paper:

> T. Khalid, A. Ahmad, I. Javed, W. Iqbal, M. A. Ali, and M. Shoaib, "Toward Reliable Few-Shot Vision–Text Fusion for Breast Cancer Histopathology Classification," 2026.

## License

The source code is released under the [MIT License](LICENSE). BreakHis, CLIP, ViT-Small and Sentence-Transformer weights are subject to their own licenses and terms of use.

## Contact

Talha Khalid — tk839587@gmail.com