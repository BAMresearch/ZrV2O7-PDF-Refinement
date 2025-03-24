
# Structural Refinement of ZrV₂O₇ with Negative Thermal Expansion Using Pair Distribution Function Analysis

Author: Tomasz Stawski (tomasz.stawski@bam.de)
Testing and Implementation: Aiste Miliute (aiste.miliute@bam.de)
Version: 1.0.17
License: MIT License

## Overview

This repository contains Python scripts specifically developed for structural refinement of Zirconium Vanadate (ZrV₂O₇), a material known for its negative thermal expansion (NTE). The scripts implement Pair Distribution Function (PDF) analysis to refine crystal structures directly from experimental X-ray diffraction (XRD) data. The refinement workflow is built around the DiffPy-CMI library, enhanced with custom functionalities tailored for ZrV₂O₇ and similar oxide materials.

The primary capabilities and key technical details of the scripts include:

- **Rigid-body Constraints and Connectivity-based Refinement:**  
  Incorporates physically-informed rigid-body restraints based on well-defined polyhedral connectivity. This includes explicit constraints on bond lengths and angles within ZrO₆ octahedra and VO₄ tetrahedra, ensuring structural parameters remain physically realistic and chemically sensible.

- **Adaptive Space-group Symmetry Switching:**  
  Provides functionality to easily transition between different space-group settings (e.g., from higher symmetry groups such as Pa-3 to lower symmetry groups like P213, P23, and ultimately P1). This capability allows the systematic exploration of potential symmetry-breaking structural distortions or subtle symmetry variations typical of NTE materials.

- **Sequential and Iterative Optimization Workflow:**  
  Performs stepwise optimization of lattice parameters, atomic coordinates, atomic displacement parameters (ADPs), oxygen occupancy, and nanoscale domain sizes (using characteristic functions). Optimization is carried out using robust gradient-based minimization algorithms (e.g., L-BFGS-B), carefully managing parameter constraints and refinement order to achieve stable convergence.

- **Parallelized PDF Calculations:**  
  Implements parallel processing using Python’s multiprocessing capabilities, optimizing performance during the calculation of PDF patterns from structural models. This significantly accelerates refinement for large datasets or complex structural models.

- **Comprehensive Visualization and Statistical Analysis:**  
  Automatically generates a variety of plots and summary statistics, including:
  - Observed vs. calculated PDF curves.
  - Difference plots (residuals).
  - Histograms of bond length distributions for Zr–O and V–O bonds.
  - Detailed bond-angle distributions for critical polyhedral linkages (O–Zr–O, O–V–O, Zr–O–V, V–O–V).
  - Export of refined crystal structures as CIF files for further analysis or reporting.

- **Detailed Logging and Reporting:**  
  Automatically logs each refinement step, including parameter adjustments, convergence criteria, refinement ranges, and space-group transitions. This functionality ensures reproducibility and facilitates detailed analysis of refinement pathways.

These scripts are specifically tailored to address the structural complexities inherent in NTE materials, providing comprehensive and reliable structural refinements optimized for ZrV₂O₇ and similar complex oxide systems.


## Requirements and Installation

The provided scripts require specific software and Python package versions to function correctly. Please carefully follow the instructions below.

### Software Requirements

| Package        | Version Requirement |
|----------------|---------------------|
| Python         | `3.7.x`             |
| DiffPy-CMI     | `3.0.0`             |
| NumPy          | `>=1.18`            |
| SciPy          | `>=1.4`             |
| Pandas         | `>=1.0`             |
| Matplotlib     | `>=3.1`             |
| Seaborn        | `>=0.11`            |
| tqdm           | `>=4.0`             |
| psutil         | `>=5.7`             |

> **Important:**  
> The scripts strictly require Python 3.7 due to compatibility constraints with DiffPy-CMI. It is strongly recommended to create a dedicated Python 3.7 environment to prevent conflicts.

---

### Creating a Dedicated Python 3.7 Environment

#### Option A: Using Conda (recommended)

To set up a new Conda environment, execute:

```bash
conda create -n diffpy python=3.7
conda activate diffpy


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

