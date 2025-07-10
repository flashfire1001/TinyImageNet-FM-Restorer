# TinyImageNet - FM - Restorer

A small demo project that uses Flow Matching to restore (denoise/reconstruct) images from the TinyImageNet dataset.

## Overview
This is a minimal PyTorch implementation demonstrating how Flow Matching can model and restore low-resolution images.  

Primarily for educational and experimental purposes.

## Quick start

```bash
git clone https://github.com/flashfire1001/TinyImageNet-FM-Restorer.git
cd TinyImageNet-FM-Restorer
pip install -r requirements.txt
python train.py      # Train the model
python sample.py     # Generate restored images
```