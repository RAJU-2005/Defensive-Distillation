# Advanced Topics in Deep Learning (ATDL) — Assignment 2
## Defensive Distillation & Adversarial Machine Learning on ResNet-34 / CIFAR-10

---

### Project Overview

This repository contains the complete, self-contained implementation, experimental analysis, and academic research report for **ATDL Assignment 2**. All code logic—including dataset parsing, the ResNet-34 model adaptation, defensive distillation training with $T^2$ scaling, untargeted white-box FGSM and PGD-20 attacks, gradient norm profiling, and external unseen test image evaluations—is implemented **100% directly inside the Jupyter Notebook lab manuals (`.ipynb`)**. There are no external `.py` module dependencies.

All models were trained from scratch for **50 epochs** per model on an NVIDIA GeForce RTX 2050 GPU (CUDA 12.1).

---

### Submission Directory Structure

```text
ATDL_Assignment2_Submission/
│
├── README.md                                       # Comprehensive project manual and guide (this document)
├── requirements.txt                                # Python dependencies specification
├── assignment_requirements_checklist.csv           # Itemized verification audit checklist (18/18 complete)
│
├── checkpoints/                                    # PyTorch model weights and training metadata
│   ├── baseline_resnet34.pth                       # ResNet-34 Baseline B weights (50 epochs)
│   ├── teacher_resnet34.pth                        # ResNet-34 Teacher M1 weights (50 epochs)
│   └── student_resnet34.pth                        # ResNet-34 Student M2 weights (50 epochs)
│
├── notebooks/                                      # Standalone, 100% Self-Contained Jupyter Notebooks
│   ├── ATDL_Assignment2_Question1_Lab_Manual.ipynb # Complete Question 1 manual with pre-rendered outputs
│   ├── ATDL_Assignment2_Question2_Lab_Manual.ipynb # Complete Question 2 manual with pre-rendered outputs
│   └── ATDL_Assignment2_Unseen_Image_Evaluation_Lab_Manual.ipynb # Unseen image manual with pre-rendered outputs
│
├── outputs/                                        # Experimental outputs, figures, and CSV data
│   ├── question1/                                  # Question 1 deliverables
│   │   ├── q1_clean_metrics_table.csv              # Test accuracy, test loss, and training times
│   │   ├── training_loss_curves.png                # Comparative 50-epoch loss curves
│   │   ├── baseline_history.json                   # Epoch-by-epoch loss and accuracy metrics
│   │   ├── teacher_history.json                    # Epoch-by-epoch loss and accuracy metrics
│   │   └── student_history.json                    # Epoch-by-epoch loss and accuracy metrics
│   │
│   ├── question2/                                  # Question 2 deliverables
│   │   ├── adversarial_accuracy_table.csv          # Comprehensive accuracy across models and attacks
│   │   ├── adversarial_summary_pivot.csv           # Clean vs FGSM vs PGD summary table
│   │   ├── accuracy_vs_epsilon.png                 # Accuracy vs perturbation budget graph
│   │   ├── adversarial_samples_student_eps8.png    # 5-sample visual grid (Clean, FGSM, PGD, Perturbations)
│   │   ├── gradient_norms_comparison.csv           # Input gradient L2 norm measurements
│   │   └── adversarial_eval_results.json           # Raw evaluation logs
│   │
│   └── unseen_images/                              # Unseen image evaluation deliverables
│       ├── unseen_evaluation_predictions.csv       # Per-image true label, predicted labels, confidence
│       ├── unseen_overall_summary.csv              # Overall accuracy and confidence metrics
│       ├── unseen_per_class_accuracy.csv           # Per-class accuracy comparison
│       ├── classification_report_*.txt             # Precision, recall, and F1-score reports
│       ├── confusion_matrix_*.png                  # Confusion matrix heatmaps (Baseline, Teacher, Student)
│       ├── unseen_model_accuracy_comparison.png    # Bar chart comparing overall accuracy
│       ├── unseen_per_class_accuracy_comparison.png# Grouped bar chart comparing per-class accuracy
│       ├── unseen_representative_grid.png          # 10-class representative visual grid
│       ├── unseen_agreement_analysis.csv           # Consensus agreement metrics
│       └── unseen_agreement_examples.png           # Visual grid of unanimous model agreements
│
├── reports/                                        # Detailed Final Academic Report
│   ├── ATDL_Assignment2_Final_Report.md            # Comprehensive Markdown report with LaTeX derivations
│   ├── ATDL_Assignment2_Final_Report.docx          # Publication-grade Microsoft Word document
│   └── ATDL_Assignment2_Final_Report.pdf           # Printable PDF report with tables and embedded figures
│
└── validation/                                     # Verification logs and audit trail
    └── execution_and_validation_summary.md         # Comprehensive test summary and compliance verification
```

---

### Notebook Architecture & Self-Containment

Every Jupyter Notebook is completely self-contained and independently executable:
* **No External `.py` Imports:** All dataset classes (`CIFAR10RawDataset`), neural network models (`ResNet34CIFAR`), attack functions (`fgsm_attack`, `pgd_attack`), gradient norm profilers, and evaluation routines are implemented directly in notebook code cells.
* **Pre-Rendered Outputs:** Each notebook was executed top-to-bottom using `NotebookClient`, embedding all tables, metrics, loss curves, and visual inspection grids directly into the notebook JSON.
* **Intelligent Checkpoint Handling:** Each notebook automatically detects existing trained checkpoints in `checkpoints/`, allowing top-to-bottom re-execution in under 30 seconds while providing full visibility into the training and evaluation code.

---

### Key Research Questions & Deliverables

#### Question 1: Defensive Distillation Implementation & Mathematical Formulation
* **Baseline Model ($B$):** Trained on hard ground-truth labels at $T = 1$ for 50 epochs (**84.62%** clean accuracy).
* **Teacher Model ($M_1$):** Trained on hard ground-truth labels at $T = 20$ for 50 epochs (**82.35%** clean accuracy).
* **Student Model ($M_2$):** Distilled from Teacher $M_1$ using soft probability targets at $T = 20$ with $T^2$ scaling for 50 epochs (**81.90%** clean accuracy).
* **Derivation of $T^2$ Multiplier:** We prove through first-order Taylor series expansion that logit gradients under unscaled distillation attenuate as $\mathcal{O}(1/T^2)$. At $T = 20$, gradients are suppressed by a factor of 400. Multiplying by $T^2$ cancels this factor and restores gradient parity with standard cross-entropy.

#### Question 2: Adversarial Robustness & Gradient Masking Analysis
* **Threat Models:** Untargeted white-box attacks conducted at inference temperature $T = 1$ in raw $[0, 1]$ pixel space with internal normalization:
  * Fast Gradient Sign Method (FGSM, single-step).
  * Projected Gradient Descent (PGD-20 with uniform random restart, step size $\alpha = 2/255$).
* **Budgets Evaluated:** $\epsilon \in \{0, 2/255, 4/255, 8/255, 16/255\}$.
* **Verification of $\epsilon = 0$:** At $\epsilon = 0$, accuracy matches clean test accuracy exactly (84.62% for Baseline, 82.35% for Teacher, 81.90% for Student).
* **Failure of Defensive Distillation:** Under PGD-20 at $\epsilon = 8/255$, Student $M_2$ accuracy collapses from **81.90% down to 12.60%**, and at $\epsilon = 16/255$ falls to **5.40%** (below random guessing).
* **Obfuscated Gradients:** Measuring input gradient norms at $T = 1$ demonstrates comparable magnitudes between Student ($\|\nabla_x \mathcal{L}\|_2 = 0.0402$) and Baseline ($\|\nabla_x \mathcal{L}\|_2 = 0.0418$). Defensive distillation merely causes gradient underflow at training temperature, providing no genuine adversarial margin expansion.

#### Supplementary Evaluation: External Unseen Test Images (200 Images)
* **Dataset:** 200 real-world external images across 10 balanced classes (20 per class) from `unseen_test_images/`.
* **Accuracy:** Baseline $B$ achieved **54.50%**, Teacher $M_1$ achieved **46.50%**, and Student $M_2$ achieved **45.50%**.
* **Model Consensus:** All three models agreed on **68.5%** of unseen samples, confirming consistent semantic feature retention across distillation despite domain shift.

---

### Quantitative Summary Tables

#### Clean Performance Summary (50 Epochs)
| Model | Training Targets | Temperature ($T$) | Epochs | Training Time (min) | Clean Test Loss | Clean Test Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline ($B$)** | Hard GT Labels | 1 | 50 | 64.8 min | 0.4932 | **84.62%** |
| **Teacher ($M_1$)** | Hard GT Labels | 20 | 50 | 63.2 min | 0.5481 | **82.35%** |
| **Student ($M_2$)** | Teacher Soft Targets | 20 | 50 | 88.5 min | 0.5620 | **81.90%** |

#### Adversarial Accuracy Across Perturbation Budgets ($\epsilon$)
| Perturbation ($\epsilon$) | Baseline B (FGSM) | Baseline B (PGD-20) | Teacher $M_1$ (FGSM) | Teacher $M_1$ (PGD-20) | Student $M_2$ (FGSM) | Student $M_2$ (PGD-20) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\epsilon = 0$ (Clean)** | **84.62%** | **84.62%** | **82.35%** | **82.35%** | **81.90%** | **81.90%** |
| **$\epsilon = 2/255$** | 58.40% | 46.20% | 55.10% | 43.50% | 54.80% | 42.10% |
| **$\epsilon = 4/255$** | 43.80% | 24.60% | 41.20% | 22.80% | 40.50% | 21.30% |
| **$\epsilon = 8/255$** | 33.20% | 14.50% | 31.80% | 13.90% | 31.40% | **12.60%** |
| **$\epsilon = 16/255$** | 24.50% | 6.20% | 23.60% | 5.80% | 23.20% | **5.40%** |

---

### How to Run the Notebooks

1. **Activate Environment:**
   ```bash
   cd c:\Users\Admin\ATDL\Assignment_2
   .\venv\Scripts\activate
   ```
2. **Launch Jupyter Lab:**
   ```bash
   jupyter lab ATDL_Assignment2_Submission/notebooks/
   ```
3. Open any of the three lab manuals:
   - `ATDL_Assignment2_Question1_Lab_Manual.ipynb`
   - `ATDL_Assignment2_Question2_Lab_Manual.ipynb`
   - `ATDL_Assignment2_Unseen_Image_Evaluation_Lab_Manual.ipynb`
4. Run cells sequentially (`Shift + Enter`) or select **Kernel -> Restart Kernel and Run All Cells**. All cells will execute top-to-bottom and display all figures and tables.

---

### Detailed Academic Report Formats
The final academic report is available in three detailed formats in `ATDL_Assignment2_Submission/reports/`:
* **Markdown:** `ATDL_Assignment2_Final_Report.md` (complete mathematical equations in LaTeX)
* **Word Document:** `ATDL_Assignment2_Final_Report.docx` (formatted with custom table styles and embedded figures)
* **Printable PDF:** `ATDL_Assignment2_Final_Report.pdf` (typeset with professional typography and figures)
