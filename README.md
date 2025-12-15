# ENMGT-5910 Vehicle Make–Model–Year Recognition (Car Recognition Module)

## Problem Statement (Insurance Triage Context)
In auto insurance claims, early-stage triage requires fast and consistent identification of a vehicle’s **make, model, and model year** from images. This information is critical for routing claims, validating inputs, and enabling downstream automation (damage assessment, cost estimation), while reducing manual effort and inconsistency during claim intake.

This repository focuses on solving **only this first step** reliably.

---

## What This Repository Does
This repository contains the **finalized car recognition module** of a larger insurance claims pipeline.

**Input**
- A **single vehicle image**

**Output**
- Predicted **make–model–year (Top-1)**
- **Top-5 candidate predictions** with confidence scores

**Included**
- Training and inference code for a **ResNet-50–based** fine-grained classifier
- Dataset handling and evaluation logic for car recognition

---

## What This Repository Does NOT Do
To keep the codebase clean, focused, and reproducible, the following components are **intentionally excluded** from this repository:

- Damage detection (e.g., YOLO / Mask R-CNN)
- Repair cost estimation
- Claim description generation (GenAI / bilingual summaries)
- Multi-image orchestration across damage photos

Folders for these modules may exist as **placeholders only** and contain no active code.

---

## Model Choice Rationale (Why ResNet-50)
**ResNet-50 (ImageNet-pretrained)** was selected as the MVP backbone because it:

- Provides a strong and stable baseline for **fine-grained visual classification**
- Performs well under **limited data and compute constraints**
- Trains efficiently and supports rapid iteration
- Is appropriate for a production-style MVP rather than a research prototype

More advanced architectures (e.g., ViT-based models) are considered **future improvements**, but ResNet-50 best fits the current scope, timeline, and deployment realism.

---

## Dataset & Task
- **Task:** Fine-grained classification across **196 make–model–year classes**
- **Data:** 16,000+ labeled vehicle images
- **Training strategy:** ImageNet pretraining + task-specific fine-tuning
- **Augmentations:** Crop, color jitter, blur, grayscale to improve generalization

---

## Performance Summary
- **Training accuracy:** ~96–97%
- **Validation/Test Top-1 accuracy:** ~84–86% (≈85%)
- **Top-5 accuracy:** ~97%

### Why ~85% Top-1 Is Sufficient
Given the strong visual similarity across many car model years, **~85% Top-1 accuracy with ~97% Top-5 accuracy** is within the expected ceiling for CNN-based fine-grained recognition and is **sufficient for triage-level automation**, especially when combined with confidence thresholds and Top-5 fallback for downstream verification.

---

## Repository Structure (Relevant Files)
- `src/recognition/baseline_resnet.py`  
  Training pipeline for the ResNet-50 car recognition model.
- `src/recognition/infer.py`  
  Single-image inference producing Top-1 and Top-5 predictions.
- `src/utils/`  
  Placeholder for future shared utilities (intentionally empty).
- `src/description/`  
  Placeholder for future description generation module (intentionally empty).
- `data/`  
  Dataset folders and placeholders.

---

## Conceptual System Context (Future Integration)
This module is designed to plug into a larger claims workflow:

1. **Car recognition (this repo)** → make/model/year
2. **Damage detection (future)** → damage type + location
3. **Cost estimation (future)** → repair cost range (USD/MXN)
4. **Description generation (future)** → structured, bilingual claim summary

Each stage is intentionally modular to support independent validation and iterative improvement.

---

## Notes
This repository reflects the **final, stable version** of the car recognition system used for academic evaluation and sponsor review. Additional pipeline components are maintained separately to preserve clarity and scope.
