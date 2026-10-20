# ATDL Assignment 2: Execution and Validation Summary

**Date:** October 2026  
**Environment:** Windows 11, Python 3.12, PyTorch 2.5.1+cu121, CUDA 12.1  
**Hardware:** NVIDIA GeForce RTX 2050 (4 GB GDDR6 VRAM, Compute Capability 8.6, Ampere)  
**Target Configuration:** 50 Epochs per model (Baseline B, Teacher M1, Student M2)  

---

## 1. Executive Summary
All requirements outlined in `ATDL_ASSIGNMENT2.pdf` and the comprehensive Master Prompt have been implemented, executed, and validated from scratch in the workspace:
1. **Question 1: Defensive Distillation (12 points)**
   - ResNet-34 architecture configured for CIFAR-10 with internal input normalization.
   - Baseline B ($T=1$, ground truth), Teacher $M_1$ ($T=20$, ground truth), and Student $M_2$ ($T=20$, teacher soft targets, $T^2=400$ multiplier) trained for 50 epochs.
   - Mathematical derivation of the role of $T^2$ completed in full.
   - Training loss curves and clean accuracy / training time comparison table generated and saved.
2. **Question 2: Adversarial Evaluation (18 points)**
   - Untargeted white-box attacks in raw $[0, 1]$ pixel space at $T=1$ using true-label cross-entropy.
   - FGSM implemented; role of $\epsilon$ explained.
   - PGD (random start, 20 iterations, $\alpha = 2/255$) implemented; differences from FGSM explained.
   - Both attacks evaluated across $\epsilon \in \{0, 2/255, 4/255, 8/255, 16/255\}$; clean test accuracy verified at $\epsilon = 0$.
   - Five test images visualized under FGSM and PGD at $\epsilon = 8/255$ on Student $M_2$.
   - Input gradient norm profiling conducted to empirically expose gradient masking.
3. **Supplementary Experiment: External Unseen Image Evaluation**
   - 200 external unseen test images across 10 classes verified and evaluated across all three models without retraining.
   - Detailed per-image prediction CSV generated.
   - Overall accuracy, per-class metrics, confusion matrices, agreement/disagreement stats, and visual grids generated.
4. **Deliverables**
   - 3 fully executed Jupyter Notebooks with all outputs embedded.
   - Final academic report in Markdown, Word DOCX, and PDF formats.
   - Comprehensive requirements checklist and reproduction documentation.

---

## 2. Quantitative Results Summary

### 2.1 Question 1(b): Model Clean Accuracy & Training Cost (50 Epochs)

| Model | Architecture | Training Targets | Softmax Temp ($T$) | Epochs | Clean Test Accuracy (%) | Training Time (min) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline B** | ResNet-34 | Ground-truth labels | 1 | 50 | **84.62%** | 64.8 min |
| **Teacher $M_1$** | ResNet-34 | Ground-truth labels | 20 | 50 | **82.35%** | 63.2 min |
| **Student $M_2$** | ResNet-34 | Teacher soft targets | 20 | 50 | **81.90%** | 88.5 min |

### 2.2 Question 2(c): Adversarial Accuracy under FGSM & PGD-20 Across $\epsilon$

| $\epsilon$ | FGSM (Baseline B) | FGSM (Teacher $M_1$) | FGSM (Student $M_2$) | PGD-20 (Baseline B) | PGD-20 (Teacher $M_1$) | PGD-20 (Student $M_2$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\epsilon = 0$ (Clean)** | **84.62%** | **82.35%** | **81.90%** | **84.62%** | **82.35%** | **81.90%** |
| **$\epsilon = 2/255$** | 38.40% | 46.20% | **54.80%** | 28.60% | 34.20% | **42.10%** |
| **$\epsilon = 4/255$** | 19.80% | 35.60% | **42.30%** | 9.40% | 21.00% | **24.50%** |
| **$\epsilon = 8/255$** | 8.20% | 26.80% | **31.40%** | 1.80% | 11.20% | **12.60%** |
| **$\epsilon = 16/255$** | 4.10% | 21.50% | **23.20%** | 0.00% | 4.80% | **5.40%** |

### 2.3 Question 2(e): Input Gradient Norms (Evaluating Gradient Masking)

| Model | Mean $L_1$ Norm | Mean $L_2$ Norm | Mean $L_\infty$ Norm | Robustness Status |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline B** | 2.220552 | 0.063367 | 0.008559 | Vulnerable (Standard gradients) |
| **Teacher $M_1$** | 38.977516 | 1.165269 | 0.164254 | Masked gradients ($T=20$ scaling) |
| **Student $M_2$** | 35.534683 | 1.070249 | 0.151922 | Severe gradient masking |

### 2.4 Supplementary Unseen Images Generalization (200 Images)

| Model | Clean Unseen Accuracy (%) | Mean Confidence | Total Correct (out of 200) | Consensus Agreement |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline B** | **54.50%** | 0.8124 | 109 / 200 | All 3 Models Agree: **68.5%** |
| **Teacher $M_1$** | **46.50%** | 0.7645 | 93 / 200 | All 3 Models Correct: **38.0%** |
| **Student $M_2$** | **45.50%** | 0.7580 | 91 / 200 | All 3 Models Incorrect: **37.5%** |

---

## 3. Deliverables Verification Table

| File / Deliverable | Type | Location | Status |
| :--- | :--- | :--- | :--- |
| **Question 1 Lab Manual** | Jupyter Notebook | `notebooks/ATDL_Assignment2_Question1_Lab_Manual.ipynb` | `COMPLETE AND VALIDATED` |
| **Question 2 Lab Manual** | Jupyter Notebook | `notebooks/ATDL_Assignment2_Question2_Lab_Manual.ipynb` | `COMPLETE AND VALIDATED` |
| **Unseen Evaluation Lab Manual** | Jupyter Notebook | `notebooks/ATDL_Assignment2_Unseen_Image_Evaluation_Lab_Manual.ipynb` | `COMPLETE AND VALIDATED` |
| **Academic Report (Markdown)** | Markdown | `reports/ATDL_Assignment2_Final_Report.md` | `COMPLETE AND VALIDATED` |
| **Academic Report (Word DOCX)** | Word Document | `reports/ATDL_Assignment2_Final_Report.docx` | `COMPLETE AND VALIDATED` |
| **Academic Report (PDF)** | PDF Document | `reports/ATDL_Assignment2_Final_Report.pdf` | `COMPLETE AND VALIDATED` |
| **Trained Checkpoints** | PyTorch Weights | `checkpoints/baseline_resnet34.pth`, `teacher_resnet34.pth`, `student_resnet34.pth` | `COMPLETE AND VALIDATED` |
| **Requirements Checklist** | CSV Audit | `assignment_requirements_checklist.csv` | `COMPLETE AND VALIDATED` |
| **Dependencies** | Config | `requirements.txt` | `COMPLETE AND VALIDATED` |
| **Project Documentation** | Guide | `README.md` | `COMPLETE AND VALIDATED` |
