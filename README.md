# AIMNet-X2D

AIMNet-X2D is a Graph Neural Network-based model for molecular property prediction with multi-task learning capabilities, designed to scale from small to large datasets. The architecture is capable of scaling to foundation model level, allowing researchers to create their own molecular foundation models with limited compute resources or scale efficiently when more hardware is available.

**Stay tuned for our upcoming paper with detailed results and methodology!**

## Features

- Multi-task learning for molecular properties
- Size-extensive additive (SAE) normalization for energy properties
- In-memory and iterable (streaming) dataset loading options for scalability
- HDF5 support for large datasets
- Multi-GPU training support via DistributedDataParallel
- Attention-based graph pooling
- Multi-hop message passing
- Stereochemical feature inclusion
- Partial charges prediction
- Molecule embedding extraction
- Foundation model scaling capabilities

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- RDKit
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aimnet-x2d.git
cd aimnet-x2d
```

2. Create a conda environment:
```bash
conda create -n aimnet-x2d python=3.12
conda activate aimnet-x2d
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

After installation, see [USAGE.md](USAGE.md) for detailed instructions on how to run AIMNet-X2D.

## Project Structure

```
AIMNet-X2D/
├── data/                  # Directory for datasets
├── models/                # Directory for saved models
├── sample-data/           # Sample datasets for testing
├── src/                   # Source code
│   ├── config.py          # Configuration handling
│   ├── datasets.py        # Dataset processing
│   ├── hyperparameter.py  # Hyperparameter optimization
│   ├── inference.py       # Inference functionality
│   ├── main.py            # Main entry point
│   ├── model.py           # Model architecture
│   ├── training.py        # Training logic
│   └── utils.py           # Utility functions
└── requirements.txt       # Package dependencies
```


## License

[MIT License]
