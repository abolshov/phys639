# Installation Guide

This repository contains a Python environment optimized for high-energy physics (HEP) and machine learning, featuring **ROOT**, **TensorFlow**, **Uproot**, **Awkward Array**, and **JupyterLab**. 

Because of the ROOT dependency, this environment is supported on **Linux** and **macOS** (both Intel and Apple Silicon).

We use **micromamba** (a fast, lightweight alternative to conda/mamba) for environment management.

---

## 1. Install Micromamba

Choose the instructions for your Operating System:

### Option A: Linux & macOS (Quick Install Script)
Run the following command in your terminal. This downloads the official installer, configures micromamba, and initializes your shell:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

## 2. Restart/reload shell

```bash
source ~/.bashrc
```
for linux or
## 2. Restart/reload shell
```bash
source ~/.zshrc
```
for macOS

## 3. Create the environment

To create the environment, run

```bash
micromamba create -f phys639.yml -y
```
To activate it run
```bash
micromamba activate phys639
```
To verify installation, run
```python
python -c "
import ROOT
import uproot
import awkward as ak
import numpy as np
import scipy
import iminuit
import tensorflow as tf
import matplotlib

print('\n🎉 SUCCESS: All packages imported successfully!')
print(f' - NumPy version: {np.__version__} (Should be < 2.2)')
print(f' - ROOT version: {ROOT.__version__}')
print(f' - TensorFlow version: {tf.__version__}')
"
```
If want to swithc to another environment, run `micromamba deactivate` and activate another environment.
