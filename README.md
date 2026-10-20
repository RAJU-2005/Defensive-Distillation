# Defensive Distillation & Adversarial Machine Learning on ResNet-34 / CIFAR-10
## Advanced Topics in Deep Learning (ATDL) — Assignment 2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture](https://img.shields.io/badge/Backbone-ResNet--34-orange.svg)](https://arxiv.org/abs/1512.03385)
[![Benchmark](https://img.shields.io/badge/Benchmark-CIFAR--10-purple.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This repository contains the complete, reproducible academic implementation of **Defensive Distillation** (Papernot et al., 2016) and an empirical investigation into its vulnerability to **White-Box Adversarial Attacks** (Fast Gradient Sign Method [FGSM] and Projected Gradient Descent [PGD-20]) on the **CIFAR-10** benchmark and an external dataset of **200 unseen test images**.

All models are built upon the **ResNet-34** architecture tailored for $32 \times 32$ spatial resolution (with $3 \times 3$ `conv1`, identity `maxpool`, and internal normalization) and trained from scratch for **50 epochs** per model on an NVIDIA GeForce RTX 2050 GPU (CUDA 12.1).

All code logic—data parsing, model architectures, distillation training loops with analytical $T^2$ scaling, attack generation, and evaluation routines—is **100% self-contained directly inside the Jupyter Notebook lab manuals (`.ipynb`)** with pre-rendered live execution outputs.

---

## 📂 Repository Structure

```text
Defensive-Distillation/
│
├── README.md                                       # Main repository documentation & run guide
├── ATDL_ASSIGNMENT2.pdf                            # Original assignment problem specification
│
├── cifar-10-batches-py/                            # Extracted CIFAR-10 benchmark dataset
│   ├── batches.meta                                # Class label metadata (10 classes)
│   ├── data_batch_1 .. data_batch_5                # Official training batches (50,000 images)
│   └── test_batch                                  # Official test batch (10,000 images)
│
├── unseen_test_images/                             # 200 external real-world test images
│   ├── airplane/ .. truck/                         # 10 balanced class subfolders (20 images/class)
│
└── ATDL_Assignment2_Submission/                    # Complete submission directory
    ├── README.md                                   # Detailed submission manual
    ├── requirements.txt                            # Full Python dependencies
    ├── assignment_requirements_checklist.csv       # Itemized requirements audit checklist (18/18)
    │
    ├── checkpoints/                                # Pre-trained ResNet-34 model weights (50 epochs)
    │   ├── baseline_resnet34.pth                   # Baseline Model B (T=1, hard GT targets)
    │   ├── teacher_resnet34.pth                    # Teacher Model M1 (T=20, hard GT targets)
    │   └── student_resnet34.pth                    # Student Model M2 (T=20, soft distilled targets)
    │
    ├── notebooks/                                  # 100% Self-Contained Jupyter Lab Manuals
    │   ├── ATDL_Assignment2_Question1_Lab_Manual.ipynb
    │   │   # Q1: Defensive distillation, T^2 derivation, 50-epoch logs (1-50), 3-panel loss plots
    │   ├── ATDL_Assignment2_Question2_Lab_Manual.ipynb
    │   │   # Q2: FGSM & PGD-20 attacks, eps sweep, 5-sample grid, gradient norm analysis
    │   └── ATDL_Assignment2_Unseen_Image_Evaluation_Lab_Manual.ipynb
    │       # Supplementary: 200 external unseen images, confusion matrices, consensus analysis
    │
    ├── outputs/                                    # Generated figures, tables, and JSON histories
    │   ├── question1/                              # Loss curves, clean metrics table, training histories
    │   ├── question2/                              # Adversarial accuracy tables, plots, sample grids
    │   └── unseen_images/                          # Predictions CSV, per-class metrics, confusion heatmaps
    │
    ├── reports/                                    # Academic Research Final Reports
    │   ├── ATDL_Assignment2_Final_Report.md        # Comprehensive Markdown report with LaTeX derivations
    │   ├── ATDL_Assignment2_Final_Report.docx      # Formatted Microsoft Word report
    │   └── ATDL_Assignment2_Final_Report.pdf       # Print-ready publication-grade PDF report
    │
    └── validation/                                 # Verification & audit logs
        └── execution_and_validation_summary.md     # Detailed verification summary
```

---

## 🚀 Run to Go: Quickstart Instructions

Follow these step-by-step instructions to clone, set up, and run the project from scratch.

### Step 1: Clone the Repository (with Git LFS)
The model checkpoints (`checkpoints/*.pth`) are tracked using **Git LFS**. Make sure Git LFS is installed:
```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository
git clone https://github.com/RAJU-2005/Defensive-Distillation.git
cd Defensive-Distillation

# Pull large checkpoint files
git lfs pull
```

### Step 2: Set Up Python Virtual Environment
Python 3.11 or later is recommended.
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r ATDL_Assignment2_Submission/requirements.txt
```

> **Note for GPU Acceleration:** If you have an NVIDIA GPU with CUDA support:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4: Launch Jupyter Lab / Notebooks
```bash
jupyter lab
```
Navigate to `ATDL_Assignment2_Submission/notebooks/` and open any of the three self-contained lab manuals:

1. **Question 1 Lab Manual:**  
   [`ATDL_Assignment2_Submission/notebooks/ATDL_Assignment2_Question1_Lab_Manual.ipynb`](ATDL_Assignment2_Submission/notebooks/ATDL_Assignment2_Question1_Lab_Manual.ipynb)  
   *Covers:* Defensive distillation architecture, mathematical derivation of the $T^2$ multiplier, 50-epoch training progression printed from Epoch 1 to 50, and 3-panel comparative loss/accuracy plots.

2. **Question 2 Lab Manual:**  
   [`ATDL_Assignment2_Submission/notebooks/ATDL_Assignment2_Question2_Lab_Manual.ipynb`](ATDL_Assignment2_Submission/notebooks/ATDL_Assignment2_Question2_Lab_Manual.ipynb)  
   *Covers:* White-box FGSM and PGD-20 attacks, accuracy evaluation across $\epsilon \in \{0, 2/255, 4/255, 8/255, 16/255\}$, clean accuracy verification at $\epsilon = 0$, 5-sample visual grid at $\epsilon = 8/255$, input gradient norm profiling, and gradient masking analysis.

3. **External Unseen Evaluation Lab Manual:**  
   [`ATDL_Assignment2_Submission/notebooks/ATDL_Assignment2_Unseen_Image_Evaluation_Lab_Manual.ipynb`](ATDL_Assignment2_Submission/notebooks/ATDL_Assignment2_Unseen_Image_Evaluation_Lab_Manual.ipynb)  
   *Covers:* Zero-shot generalization on 200 real-world external images across 10 balanced classes, confusion matrix analysis, and multi-model consensus statistics.

*All notebooks run seamlessly from top to bottom (`Shift + Enter` or `Kernel -> Restart and Run All Cells`) and feature intelligent checkpoint detection to execute in under 30 seconds.*

---

## 📊 Summary of Experimental Results (50 Epochs)

### 1. Clean Classification Performance (CIFAR-10 Test Set)
| Model Architecture | Training Method | Temperature ($T$) | Epochs | Training Time (min) | Clean Test Loss | Clean Test Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Model ($B$)** | Standard Cross-Entropy (GT) | 1 | 50 | 64.8 min | 0.4932 | **84.62%** |
| **Teacher Model ($M_1$)** | Temperature Cross-Entropy (GT) | 20 | 50 | 63.2 min | 0.5481 | **82.35%** |
| **Student Model ($M_2$)** | Defensive Distillation ($T^2$ Soft) | 20 | 50 | 88.5 min | 0.5620 | **81.90%** |

* **Mathematical Necessity of $T^2$ Scaling:**
  $$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i} = \frac{1}{T}(p_i - q_i) \approx \frac{(z_i - \bar{z}) - (v_i - \bar{v})}{K \cdot T^2}$$
  At $T = 20$, gradients are suppressed by $20^2 = 400$. Multiplying the distillation objective by $T^2$ cancels this attenuation, restoring learning dynamics to parity with hard-target cross-entropy.

---

### 2. Adversarial Accuracy vs. Perturbation Budget ($\epsilon$)
Evaluated under white-box conditions at inference temperature $T = 1$:
| Perturbation Budget ($\epsilon$) | Baseline B (FGSM) | Baseline B (PGD-20) | Teacher $M_1$ (FGSM) | Teacher $M_1$ (PGD-20) | Student $M_2$ (FGSM) | Student $M_2$ (PGD-20) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\epsilon = 0$ (Clean)** | **84.62%** | **84.62%** | **82.35%** | **82.35%** | **81.90%** | **81.90%** |
| **$\epsilon = 2/255$** | 58.40% | 46.20% | 55.10% | 43.50% | 54.80% | 42.10% |
| **$\epsilon = 4/255$** | 43.80% | 24.60% | 41.20% | 22.80% | 40.50% | 21.30% |
| **$\epsilon = 8/255$** | 33.20% | 14.50% | 31.80% | 13.90% | 31.40% | **12.60%** |
| **$\epsilon = 16/255$** | 24.50% | 6.20% | 23.60% | 5.80% | 23.20% | **5.40%** |

* **The Fallacy of Gradient Masking:**
  Under iterative PGD-20 at $\epsilon = 8/255$, Student $M_2$ collapses to **12.60%**, and at $\epsilon = 16/255$ plummets to **5.40%** (worse than random guessing). Measuring input gradient norms ($\|\nabla_x \mathcal{L}\|_2 \approx 0.0402$ vs. $0.0418$) proves that defensive distillation merely produces **obfuscated gradients** when tested at high temperatures, but provides zero true adversarial robustness when deployed at standard inference temperature $T = 1$.

---

### 3. External Unseen Image Evaluation (200 Real Images)
| Metric | Baseline Model ($B$) | Teacher Model ($M_1$) | Student Model ($M_2$) |
| :--- | :---: | :---: | :---: |
| **Evaluated Images** | 200 / 200 | 200 / 200 | 200 / 200 |
| **Overall Accuracy** | **54.50%** | **46.50%** | **45.50%** |
| **Mean Softmax Confidence** | 78.4% | 72.1% | 71.5% |
| **Unanimous Model Agreement** | \multicolumn{3}{c|}{**68.5%** (137 / 200 images)} |
| **Model Disagreements** | \multicolumn{3}{c|}{**31.5%** (63 / 200 images)} |

* Real-world photographic domain shift causes a ~30% drop in accuracy compared to curated CIFAR-10 crops. All three models exhibit 68.5% unanimous consensus, confirming consistent feature retention across distillation.

---

## 📑 Academic Final Reports

The complete academic research report is provided in three publication-grade formats inside `ATDL_Assignment2_Submission/reports/`:
* [**Markdown Report**](ATDL_Assignment2_Submission/reports/ATDL_Assignment2_Final_Report.md): Full LaTeX mathematical formulations, embedded figures, and analytical derivations.
* [**Microsoft Word Document**](ATDL_Assignment2_Submission/reports/ATDL_Assignment2_Final_Report.docx): Professional typography with styled tables and figure captions.
* [**PDF Document**](ATDL_Assignment2_Submission/reports/ATDL_Assignment2_Final_Report.pdf): Publication-ready printable PDF report (827 KB).

---

## 📚 Academic References

1. **Papernot, N., McDaniel, P., Wu, X., Jha, S., & Swami, A.** (2016). *Distillation as a defense to adversarial perturbations against deep neural networks*. IEEE Symposium on Security and Privacy (S&P), 582–597.
2. **Hinton, G., Vinyals, O., & Dean, J.** (2015). *Distilling the knowledge in a neural network*. arXiv:1503.02531.
3. **Goodfellow, I. J., Shlens, J., & Szegedy, C.** (2014). *Explaining and harnessing adversarial examples*. International Conference on Learning Representations (ICLR).
4. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.** (2017). *Towards deep learning models resistant to adversarial attacks*. International Conference on Learning Representations (ICLR).
5. **Carlini, N., & Wagner, D.** (2017). *Towards evaluating the robustness of neural networks*. IEEE Symposium on Security and Privacy (S&P), 39–57.
6. **Athalye, A., Carlini, N., & Wagner, D.** (2018). *Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples*. International Conference on Machine Learning (ICML).
7. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep residual learning for image recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

---

## 👤 Author
* **Raju** ([@RAJU-2005](https://github.com/RAJU-2005))
