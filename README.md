
# DiffPy-based Structural Refinement Framework for PDF Data

Author: Tomasz Stawski (tomasz.stawski@bam.de)
Version: 1.0.0
License: MIT License

## Overview

This repository provides Python scripts and tools developed for structural refinement using Pair Distribution Function (PDF) analysis. It leverages the capabilities of DiffPy-CMI and optionally PDFgetX3.

Features:

- Generation of PDFs from X-ray diffraction (XRD) data.
- Advanced structural refinement using space-group constraints.
- Customizable optimization and refinement strategies.
- Comprehensive visualization and summary statistics of refined structures.

## Installation

### Requirements

Package        | Required Version
-------------- | ----------------
Python         | 3.7.x
DiffPy-CMI     | 3.0.0
PDFgetX3       | 2.1.1 (optional)
NumPy          | >= 1.18
SciPy          | >= 1.4
Pandas         | >= 1.0
Matplotlib     | >= 3.1
Seaborn        | >= 0.11
tqdm           | >= 4.0
psutil         | >= 5.7

> **Important:** DiffPy-CMI requires Python 3.7 specifically. Create a dedicated Python 3.7 environment.

### Recommended Setup (New Python 3.7 Environment)

#### Conda (Recommended)

```bash
conda create -n diffpy python=3.7
conda activate diffpy

pip install numpy>=1.18 scipy>=1.4 pandas>=1.0 matplotlib>=3.1 seaborn>=0.11 tqdm>=4.0 psutil>=5.7
```

#### virtualenv

```bash
python3.7 -m venv diffpy-env
source diffpy-env/bin/activate

pip install numpy>=1.18 scipy>=1.4 pandas>=1.0 matplotlib>=3.1 seaborn>=0.11 tqdm>=4.0 psutil>=5.7
```

### DiffPy-CMI Installation

Activate your environment, then run:

```bash
pip install diffpy-cmi==3.0.0
```

[DiffPy-CMI installation guide](https://www.diffpy.org/products/diffpycmi/)

### Optional: PDFgetX3 Installation

Requires a separate license:

```bash
pip install diffpy.pdfgetx==2.1.1
```

[PDFgetX3 details](https://www.diffpy.org/products/pdfgetx3/)

## Repository Structure

```
├── CIFs/                   # Input crystallographic structures (.cif)
├── data/                   # Diffraction or pre-calculated PDF data
├── fits/                   # Directory for output results
├── LICENSE                 # MIT License file
├── sample_refinement.py    # Main structural refinement script
├── README.md               # Documentation (this file)
└── utils/                  # Utility scripts and additional functions
```

## Running the Refinement

Activate your environment, then run:

```bash
conda activate diffpy
python sample_refinement.py
```

### Important Configuration Variables

```python
mypowderdata = 'data/my_diffraction_data.dat'
composition = 'O7 V2 Zr1'

ciffile = {
    'example_structure.cif': ['P213', False, (1, 1, 1)]
}

anisotropic = False
unified_Uiso = True
```

## Results and Outputs

- Summary reports (.txt files in `fits/`)
- Visualizations (plots, distributions as `.png`, `.pdf`)
- CIF files (exported refined structures)

## License

MIT License (see LICENSE)

## Citation and Acknowledgments

Cite relevant DiffPy and PDFgetX3 resources from [DiffPy.org](https://www.diffpy.org/).

Special thanks to the DiffPy and DANSE Diffraction groups.

## Support and Contact

Tomasz Stawski  
tomasz.stawski@bam.de
tomasz.stawski@gmail.com  

