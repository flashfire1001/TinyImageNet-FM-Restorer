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


## 🗂️ **Project Structure**

```
TinyImageNet-FM-Restorer/
│
├── data/                   # Datasets, including TinyImageNet
│   ├── tinyimagenet/       # TinyImageNet data
│   ├── processed/          # Preprocessed or augmented data
│   └── README.md           # Dataset info and instructions
│
├── cores/                  # Machine learning model definitions
│   ├── sample/
│       ├── corruption and the 
│   ├── unet.py           	# unet architectures for feature extraction
│   ├── path.py    			# Path or the ode
│   ├── simulator.py        # simulator for the model
│   ├── visual.py        	# visualization functions
│   ├── metrics.py        	# functions that implement evaluation by different metrics
│   └── utils.py            # HelperFunctions (save models/load pre-trained models)
│
├── notebooks/              # Jupyter Notebooks (for exploration, analysis, etc.)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_flow_matching.ipynb
│   └── 03_evaluation.ipynb
│
├── scripts/                # Python scripts for training, inference, etc.
│   ├── train.py            # Script to train the model
│   ├── evaluate.py         # Evaluate model performance by different metric
│   ├── denoise.py          # Denoising / restoring images
│   └── preprocess.py       # Preprocessing data before training
│
├── logs/                   # Logs for training, model checkpoints
│   ├── training_logs/      # Logs from training sessions
│   └── checkpoints/trial01        # Model checkpoints of different trials
│
├──experiments/
│    ├── fm_baseline.yaml
│    ├── unet_denoising.yaml
│    └── trial_log.txt
│
├── requirements.txt        # List of Python dependencies
├── README.md               # Project overview and instructions
└── config.yaml             # Configuration file for hyperparameters, paths, etc.
```

------

### 📄 **File Descriptions**

- **`data/`**: Contains all dataset-related files.
  - **`tinyimagenet/`**: Raw images and annotations from TinyImageNet.
  - **`processed/`**: Preprocessed images (e.g., resized, normalized) ready for training.
- **`models/`**: Stores model architecture and utility code.
  - **`flow_matching.py`**: The core implementation of the **Flow Matching** approach for image restoration.
  - **`resnet.py`**: A backbone model for feature extraction, possibly used in Flow Matching.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis, experiments, and results visualization.
  - **`01_data_preprocessing.ipynb`**: Notebook for exploring and preprocessing TinyImageNet data.
  - **`02_flow_matching.ipynb`**: Experiment with Flow Matching algorithms and their effect on image restoration.
  - **`03_evaluation.ipynb`**: Evaluation of model performance using different metrics (PSNR, SSIM, etc.).
- **`scripts/`**: Python scripts for execution tasks.
  - **`train.py`**: A script to train the Flow Matching model using the TinyImageNet dataset.
  - **`evaluate.py`**: A script to evaluate the model's performance on denoising/reconstruction tasks.
  - **`denoise.py`**: A script to run the trained model on noisy input and restore the image.
  - **`preprocess.py`**: A script to preprocess data before it is fed into the model.
- **`logs/`**: Contains logs from training and checkpoints.
  - **`training_logs/`**: Logs that track the progress and loss during training.
  - **`checkpoints/`**: Saved model checkpoints to resume or reload the model.
- **`requirements.txt`**: Lists all the Python libraries required to run the project (e.g., TensorFlow, PyTorch, etc.).
- **`README.md`**: Provides an overview of the project, setup instructions, and usage.
- **`config.yaml`**: Contains configuration settings for training, hyper-parameters, and data paths.



