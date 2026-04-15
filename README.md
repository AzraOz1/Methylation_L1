# Methylation_L1
Early prototype of a PyTorch pipeline to classify OCD vs control status using OXTR methylation features
Current version uses synthetic data based on patterns described in the literature while real data collection is in progress.


## Current status

The current version uses a synthetic dataset of 200 samples generated from patterns described in the literature. This was done to prototype the workflow before the final real dataset is gathered and organized.
Literature used: Bey K., Campos-Martin R., Klawohn J., Reuter B., Grutzmann R., Riesel A., et al. (2022). Hypermethylation of the oxytocin receptor gene (OXTR) in obsessive-compulsive disorder: further evidence for a biomarker of disease and treatment response. Epigenetics 17, 642–652. 10.1080/15592294.2021.1943864 [DOI] [PMC free article] [PubMed] [Google Scholar]

This is therefore it is not a finalized clinical model.

## Features used

Each sample represents one individual and includes:

1) 10 OXTR CpG methylation features
2) Age (normalized)
3) Sex (binary encoded)

Total input features: 12

## ML pipeline overview

The pipeline was built in the following stages:

### 1. Data generation
Since the final dataset is still being assembled, the current script generates synthetic data based on methylation patterns reported in the literature.

### 2. Dataset construction
The synthetic features and labels are wrapped into a custom PyTorch `Dataset`, where features are converted to float tensors and labels are converted to integer class tensors.

### 3. Model architecture
The classifier is a feedforward neural network with:
- input layer: 12 features
- hidden layer 1: 32 units + ReLU + dropout
- hidden layer 2: 16 units + ReLU + dropout
- output layer: 2 logits (Control vs. OCD)

### 4. Training
The model is trained using CrossEntropyLoss, Adam optimizer, mini-batch training, learning rate scheduling.

the final dataset is available


## Tech stack

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
