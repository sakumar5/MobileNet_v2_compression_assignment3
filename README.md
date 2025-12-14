# MobileNet_v2_compression_assignment3
training MobileNet-v2 on CIFAR-10 and then applying model compression techniques to reduce model size while retaining accuracy.

 ---
 ## Model Training and Post-Training Quantization on CIFAR
 
Name       : Sanjeev Kumar
Roll No:   : CS24M533 
Assignment : Assignment 3  

Implementation Of :  **CS6886W Assignment 3**, 
focus point       : **training a convolutional neural network and applying post-training quantization (PTQ)** 
Evaluation        : Accuracy-compression trade-offs

Implementation is done using **PyTorch** and  Includes utilities for training, evaluation, quantization, and compression analysis.
---

## Project Overview

The objectives of this assignment are:

- Train a **MobileNetV2-based model** on the CIFAR dataset  
- Apply **post-training quantization (PTQ)** techniques  
- Evaluate the impact of quantization on:
  - Model accuracy  
  - Model size and compression ratio  
- Provide a modular and reproducible training and evaluation pipeline  

---

## Repository Structure

```
CS6886W_A3/
|
|-- cs24m533/
|  |-- config.py                  # Centralized configuration and argument parsing
|  |-- data.py                    # Dataset loading and preprocessing (CIFAR)
|  |-- train.py                   # Model training pipeline
|  |-- eval.py                    # Evaluation of trained and quantized models
|  |-- measure_compression.py     # Model size & compression ratio analysis
|  |-- utils.py                   # Helper functions (seeding, metrics, schedulers)
|  |
|  |-- models/
|  |  |-- mobilenetv2_cifar.py   # MobileNetV2 adapted for CIFAR
|  |  |-- __init__.py
|  |
|  |-- compression/
|  |  |-- activation_wrappers.py # Activation quantization wrappers
|  |  |-- quant_ops.py           # Custom quantization operators
|  |  |-- ptq_utils.py           # Post-training quantization utilities
|  |  |--__init__.py
| 
|-- CS6886W___Assignment_3.pdf      # Assignment description (provided)
|-- README.md                      # Project documentation
```

---

## Environment Setup

### Requirements
- Python >= 3.9  
- PyTorch  
- torchvision  
- numpy  
- tqdm  
- wandb (optional, for experiment tracking)

### Installation
```bash
pip install torch torchvision numpy tqdm wandb
```
---

##  Training the Model
To train the baseline (floating-point) model:

```bash
python cs24m533/train.py
```

Key features:
- Configurable hyperparameters via `config.py`
- Reproducibility through fixed random seeds
- Training metrics logged per epoch
- Optional Weights & Biases (wandb) integration
---

## Model Evaluation

To evaluate a trained model:

```bash
python cs24m533/eval.py
```

This script:
- Loads trained model checkpoints  
- Computes accuracy on the test set  
- Supports evaluation of both **FP32 and quantized models**

---

##  Post-Training Quantization (PTQ)

Quantization functionality is implemented in the `compression/` module and includes:

- Weight quantization  
- Activation quantization via wrappers  
- Calibration utilities for PTQ  

Quantized evaluation is integrated into the evaluation pipeline.

---

## Compression Analysis

To measure model size and compression ratio:

```bash
python cs24m533/measure_compression.py
```

This script reports:
- Original (FP32) model size  
- Quantized model size  
- Compression ratio  

---

## Configuration

All training and quantization parameters are managed via:

```
cs24m533/config.py
```

Command-line arguments override default configuration values.

---

##  Reproducibility

- Fixed random seeds are used for deterministic behavior  
- Modular design allows easy experimentation  
- Clear separation of training, evaluation, and quantization logic  

---

##  Assignment Context

This repository is submitted as part of **CS6886W - Assignment 3**.  
The provided `CS6886W___Assignment_3.pdf` contains the full problem statement and evaluation criteria.

the submitted report is **cs24m533_Sanjeev_kumar_assignment_CS6886W_report.pdf**
---

##  Author
- **Name:** Sanjeev Kumar 
- **Roll Number:** cs24m533  
- **Course:** CS6886W  

---
