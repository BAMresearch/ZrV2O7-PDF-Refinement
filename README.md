
# Structural Refinement of ZrV₂O₇ with Negative Thermal Expansion Using Pair Distribution Function Analysis

Author: Tomasz Stawski (tomasz.stawski@bam.de)
Testing and Implementation: Aiste Miliute (aiste.miliute@bam.de)
Version: 1.0.17
License: MIT License

![Animated structure of ZrV₂O₇](./images/structure.gif)

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

---

## Requirements and Installation

This section outlines software installation using Anaconda, with required and optional software.

### Software Requirements

- Python 3.7 (strictly required for compatibility with DiffPy-CMI)
- DiffPy-CMI `3.0.0`, [DiffPy-CMI](https://www.diffpy.org/products/diffpycmi/index.html).
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn
- tqdm
- psutil


> **Important:**  
> Python 3.7 is strictly required for compatibility with DiffPy-CMI. Creating a dedicated Python 3.7 environment is strongly recommended.

---

### Creating a Dedicated Anaconda Python 3.7 Environment

Create and activate a new Anaconda environment:

```bash
conda create -n diffpy python=3.7
conda activate diffpy
```

---

### Installing DiffPy-CMI (Anaconda only)

Add the DiffPy Anaconda channel and install DiffPy-CMI:

```bash
conda config --add channels diffpy
conda install diffpy-cmi
```

---

### Optional but Highly Recommended: PDFgetX3

PDFgetX3 is strongly recommended for direct PDF generation from raw XRD data but requires a separate license.

**Note:**  
If PDFgetX3 is not installed, scripts must be manually adjusted to directly load pre-generated PDF/g(r) data from CSV files. 
For more information about PDFgetX3, visit [DiffPy PDFgetX3](https://www.diffpy.org/products/pdfgetx.html).

---
## Usage

This section describes how to set up your inputs and explains step-by-step the refinement procedure as implemented in `sample_refinement.py`.

---

### Project and Data Inputs

All inputs are specified at the end of `sample_refinement.py`. Adjust these variables according to your experimental data and desired refinement approach.

Example input definitions from the script:

```python
# Project and directories
project_name = 'OneZirconiumVanadate105C/'
xrd_directory = 'data/'        # Directory containing diffraction or PDF data
cif_directory = 'CIFs/'        # Directory with structure CIF files
fit_directory = 'fits/'        # Directory for refinement outputs

# Experimental XRD data filename
mypowderdata = 'PDF_ZrV2O7_061_105C_avg_246_265_00000.dat'

# Chemical composition for PDF generation
composition = 'O7 V2 Zr1'

# Structural phases to refine (CIF files and corresponding symmetry)
ciffile = {'98-005-9396_ZrV2O7.cif': ['Pa-3', True, (1, 1, 1)]}

# Instrument parameters (from calibration or instrument profile)
qdamp = 2.70577268e-02     # Instrumental damping factor
qbroad = 2.40376789e-06    # Instrumental broadening factor

# Atomic displacement parameters (ADPs)
anisotropic = False        # Use isotropic ADPs (True if anisotropic ADPs desired)
unified_Uiso = True        # Same Uiso values for atoms of the same element

# Space group symmetry offset (usually [0,0,0])
sgoffset = [0.0, 0.0, 0.0]

# PDF calculation and refinement parameters
myrange = (0.0, 80)        # Range for PDF calculation (in Å)
myrstep = 0.05             # Step size for r-axis in PDF

# Optimization convergence criteria
convergence_options = {'disp': True}
```


---
## Usage and Fitting Procedure

This section describes the detailed workflow implemented in `sample_refinement.py` for structural refinement of Zirconium Vanadate (ZrV₂O₇) from PDF data. The script is capable of refining single-phase structures and can be adapted for multi-phase systems with additional adjustments.

---

### Overview

The refinement procedure involves several systematic stages:

1. **Generating the experimental PDF** from X-ray diffraction data.
2. **Creating PDF contributions** linked to structural models defined by CIF files.
3. **Setting up a refinement recipe** with constraints based on the initial symmetry and rigid-body requirements.
4. **Sequential refinement** in several stages, progressively lowering the space-group symmetry.
5. **Applying rigid-body constraints** on bond lengths and angles to ensure physically meaningful structures.
6. **Collecting and visualizing results**, including partial PDFs, refined CIF files, bond-length distributions, and angle statistics.

---

### Detailed Refinement Steps

Each refinement stage follows this structured approach:

- **Space-group symmetry adjustment:**  
  Change structural symmetry from high (Pa-3) to lower symmetry settings (P213, P23, P1).

- **Rigid-body constraints:**  
  Apply and adjust constraints on bond lengths and angles.

- **Sequential parameter refinement:**  
  Parameters refined in each step include lattice parameters (`lat`), scale factors (`scale`), particle size (`psize`), peak shape (`delta2`), atomic displacement parameters (`adp`), and atomic positions (`xyz`).

**Specific refinement steps:**

| Step | Space Group | Bond Constraints (σ) | Angle Constraints (σ) | PDF range (Å) | Purpose |
|------|-------------|----------------------|-----------------------|---------------|---------|
| 0    | Pa-3        | 0.001                | 0.001                 | 1.5–27        | Initial high-symmetry refinement |
| 1    | Pa-3        | 0.0001               | 0.0001                | 1.5–27        | Refine with tighter constraints |
| 2    | P213        | 0.001                | 0.001                 | 1.5–27        | Test response to reduced symmetry |
| 3    | P23         | 0.001                | 0.001                 | 1.5–27        | Further symmetry exploration |
| 4    | P23         | 0.0001               | 0.0001                | 1.5–27        | Precise refinement at lower symmetry |
| 5    | P1          | 0.001                | 0.001                 | 1.5–27        | Lowest symmetry flexibility |
| 6    | P1          | 0.0001               | 0.0001                | 1.5–27        | Final refinement under strictest constraints |

After completing these steps, the script extrapolates the final refined model across the full PDF range (0–80 Å) for comprehensive evaluation.

---

## Multi-Phase Refinements (Optional)

The provided refinement script supports simultaneous fitting of multiple structural phases. The basic functionality—such as PDF generation, scaling, and individual phase contributions—is fully operational. However, the use of rigid-body constraints (bond lengths, angles, dihedrals) in multi-phase scenarios has not been thoroughly tested and may introduce unexpected behaviors. Additional verification is recommended if rigid-body constraints are applied across multiple phases.

### How to Define Multi-Phase CIF Input

Multi-phase refinements are configured by modifying the `ciffile` dictionary in your script. The dictionary format is as follows:

```python
ciffile = {
    'Phase1_filename.cif': ['SpaceGroup1', periodic1, (nx1, ny1, nz1)],
    'Phase2_filename.cif': ['SpaceGroup2', periodic2, (nx2, ny2, nz2)],
    # Add additional phases as needed
}

```
---

## Rigid-Body Constraints Implementation

Rigid-body constraints ensure physically meaningful refinements by controlling bond lengths, bond angles, and optionally dihedral angles. These constraints are particularly important for complex structures like ZrV₂O₇ to avoid unphysical configurations.

### Step-by-Step Procedure

The script implements rigid-body constraints through the following sequential steps:

### 1. **Calculation of Bond Vectors**  
- The script first identifies relevant polyhedral units (ZrO₆ and VO₄) and computes bond vectors within these units, applying predefined distance cutoffs.

### 2. **Identification of Bond Pairs**  
- Using the calculated bond vectors, bond pairs (Zr–O, V–O, and O–O) are determined for each polyhedron, considering symmetry and periodic boundary conditions.

### 3. **Angle and Dihedral Identification**  
- Bond angles (e.g., O–Zr–O, O–V–O, Zr–O–V, V–O–V) are identified based on connectivity. Optionally, dihedral angles involving four-atom combinations are also determined, if specified in the script.

### 4. **Constraint Expression Generation**  
- Mathematical expressions describing bond lengths and angles in terms of atomic coordinates are generated dynamically. These expressions form the basis of the constraints applied during refinement.

### 5. **Classification and Application of Constraints**  
- Bond constraints are categorized and applied with varying strictness:
  - **Normal Bonds:** Constraints typically applied with standard deviation (`σ`) around 0.001 to 0.0001.
  - **Shared Bonds (e.g., V–O–V):** More strictly constrained due to their structural significance, typically with very tight σ (e.g., 1e-8).
  - **Edge Bonds:** Bonds near unit cell boundaries are constrained tightly (σ ~1e-7).
  - **Problematic Bonds:** Bonds outside acceptable lengths (<1.6 Å or >1.9 Å for V–O bonds) receive stricter constraints.

- Angle and dihedral constraints follow a similar categorization, typically constrained with a narrow tolerance (~±1–2°).

### 6. **Dynamic Updating (Adaptive Constraints)**  
- Optionally, constraints can be recalculated dynamically (adaptive constraints) after each refinement step to account for structural changes, although this feature is disabled (`adaptive=False`) by default.

### 7. **Final Cleanup of Constraints**  
- Constraints that become irrelevant (e.g., due to symmetry changes or refinement progress) are automatically removed from subsequent refinement steps.

---

### How to Set Rigid-Body Constraints in the Script

The strength and type of constraints are controlled via these parameters:

```python
constrain_bonds = (True, 0.001)      # Enable bond constraints with σ = 0.001
constrain_angles = (True, 0.001)     # Enable angle constraints with σ = 0.001
constrain_dihedrals = (False, 0.001) # Dihedral constraints disabled by default
```

## Output Data Structure

The refinement results generated by the script are systematically organized in timestamped directories under the `fits/` directory. This clear structure ensures easy navigation and analysis of your refinement outputs.

## Performance Note

The unit cell of the ZrV₂O₇ structure considered in this refinement is large, containing **1080 atoms**. Consequently, the number of fitted parameters increases significantly as the refinement progresses and symmetry constraints are progressively relaxed to lower space groups (from Pa-3 to P1).

As a practical reference, refining the full series of symmetry reductions and rigid-body constraints on an **AMD Ryzen 7840U** processor, with approximately **80% CPU utilization**, typically takes about **36 hours** to complete all refinement steps. Users should consider these performance implications when planning refinements.


## License

MIT License (see LICENSE)


## Support and Contact

Tomasz Stawski  
tomasz.stawski@bam.de  
tomasz.stawski@gmail.com







