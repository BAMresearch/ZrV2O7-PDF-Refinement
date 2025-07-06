"""
Created on Thu Nov 23 12:39:22 2023

@author: Tomasz Stawski
tomasz.stawski@gmail.com
tomasz.stawski@bam.de

# =============================================================================
# DESCRIPTION:
# This script demonstrates a refinement procedure for ZrV2O7-based structures.
# Users can modify space group symmetry mid-stream to see how the structure
# adapts to different symmetry constraints. Rigid-body constraints (e.g. bond lengths,
# angles) can be applied or removed as needed.
#
# The steps in the script are:
#   1) Configure and generate the experimental PDF
#   2) Create the PDFContribution and link it to one or more CIF structures
#   3) Build a FitRecipe with default or optional constraints
#   4) Refine in multiple stages
#   5) Optionally change space groups and re-apply constraints
#   6) Collect results, partial PDFs, final fits, etc.
"""
# =============================================================================
# load libraries and settings 
# =============================================================================

# =============================================================================
# Set the working directory to the current directory.
# This ensures that all relative paths work as expected.
from os import chdir, getcwd
wd = getcwd()
chdir(wd)

# =============================================================================
# Configure matplotlib for plotting
# Set the default figure size to 5 x 3.75 inches for consistency.
import matplotlib
matplotlib.rc('figure', figsize=(5, 3.75))
from matplotlib.pyplot import subplots
import seaborn as sns

# =============================================================================
# Import data management packages
# These packages are essential for data manipulation, numerical calculations, 
# and object serialization.
import copy         # For deep and shallow copying of objects.
import numpy as np  # Numerical operations and arrays.
import pandas as pd # Data manipulation and analysis.
import os           # Operating system interface.
import io           # Working with streams
import contextlib   # Context manager
import pickle       # For serializing and deserializing Python objects.
from collections import namedtuple # For creating refinement steps object for easy access of parameters 
from datetime import datetime  # for time related operations
from itertools import combinations  # Combinatorial functions

# =============================================================================
# Import PDF generation library (PDFgetX3) for XRD data processing
# Provides functions to extract PDFs from diffraction data.
from diffpy.pdfgetx import PDFGetter, PDFConfig, loadData

# =============================================================================
# Import diffpy libraries for structural refinement
# These include tools for defining fitting recipes, parsing data, 
# handling crystal structures, and calculating PDFs.
from diffpy.srfit.fitbase import FitRecipe, FitResults, FitContribution, Profile
from diffpy.srfit.pdf import PDFContribution, PDFGenerator, PDFParser
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.Structure import loadStructure
from diffpy.Structure import expansion
from diffpy.srreal.pdfcalculator import PDFCalculator
from pyobjcryst import loadCrystal
from diffpy.srfit.pdf.characteristicfunctions import sphericalCF, sheetCF
from diffpy.srfit.pdf import DebyePDFGenerator

# =============================================================================
# Import optimization algorithms from SciPy
# Provides a selection of optimizers, including least-squares and gradient-based methods.
from scipy.optimize import leastsq, least_squares, minimize

# =============================================================================
# Enable multi-threaded processing for parallel computations
# Uses multiprocessing and psutil to manage CPU usage efficiently.
import psutil
import multiprocessing
from multiprocessing import Pool, cpu_count

# =============================================================================
# Additional utilities for progress tracking and distance calculations
# tqdm: For progress bars
# scipy.spatial.distance: For calculating pairwise distances
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
#                              PROJECT SETUP
# =============================================================================

# =============================== Input Definitions ===========================
# Define project name and directories
project_name = 'ZirconiumVanadate25Cto25C_Rec_False/'
xrd_directory = 'data/'   # Directory containing diffraction data
cif_directory = 'CIFs/'    # Directory containing CIF files
fit_directory = 'fits/'    # Base directory for storing refinement results

# =============================== XRD Data ====================================
mypowderdata = 'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat'

# =============================== Output Setup ================================
time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
output_results_path = fit_directory + project_name + str(time_stamp) + '/'
if not os.path.isdir(output_results_path):
    os.makedirs(output_results_path)

# =============================== Phase Setup ================================
ciffile = {'98-005-9396_ZrV2O7.cif': ['Pa-3', True, (1, 1, 1)]}

# ========================== Atomic Composition ===============================
composition = 'O7 V2 Zr1'

# =============================================================================
# Global composition dictionary
# =============================================================================
# For each element:
#  • symbol            : element symbol
#  • Uiso              : initial isotropic ADP guess
#  • polyhedron_center : True if this element is the center of a coordination polyhedron
#  • polyhedron_vertex : True if this element acts as a vertex in any polyhedron
#  • cutoff            : (min,max) bond-lengths (Å) to its vertex elements (only for centers)
detailed_composition = {
    'Zr': {
        'symbol': 'Zr',
        'Uiso': 0.0065,
        'polyhedron_center': True,
        'polyhedron_vertex': False,
        'cutoff': (1.8, 2.2),
    },
    'V': {
        'symbol': 'V',
        'Uiso': 0.0100,
        'polyhedron_center': True,
        'polyhedron_vertex': False,
        'cutoff': (1.5, 2.4),
    },
    'O': {
        'symbol': 'O',
        'Uiso': 0.0250,
        'polyhedron_center': False,
        'polyhedron_vertex': True,
    },
}

# =========================== Instrumental Parameters =========================
qdamp = 2.70577268e-02
qbroad = 2.40376789e-06
qmax = 22.0

# ================= Atomic Displacement Parameters (ADPs) ====================
anisotropic = False
unified_Uiso = True

# ======================= Space Group Offset ================================
sgoffset = [0.0, 0.0, 0.0]

# ======================= PDF Generation Parameters ==========================
myrange = (0.0, 80)
myrstep = 0.05

# ================== Fitting Procedure Parameters ============================
convergence_options = {'disp': True}






###############################################################################
#                               ALL THE FUNCTIONS 
###############################################################################

# =============================================================================
# Checkpointing functions
# =============================================================================
def save_checkpoint(step, fit, cpdf, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"step": step, "fit": fit, "cpdf": cpdf}, f)


def load_checkpoint(filename="checkpoint.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["step"], data["fit"], data["cpdf"]
    else:
        return 0, None, None  # Start from scratch

# =============================================================================
# Visualisation and summary functions
# =============================================================================
def plotmyfit(cpdf, baseline=-4, ax=None):
    """
    Plot the observed, calculated, and difference (Gobs, Gcalc, Gdiff) PDFs.

    Parameters:
    - cpdf: PDFContribution object, containing observed profile + model.
    - baseline: Float, baseline offset for difference plot (default: -4).
    - ax: Matplotlib axis object, optional. If None, a new axis is created.

    Returns:
    - rv: List of matplotlib line objects for the plots.
    - df: Pandas DataFrame with columns ['x', 'yobs', 'ycalc', 'ydiff'].
    """
    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # Pull observed G(r) from cpdf
    x = cpdf.profile.x
    yobs = cpdf.profile.y

    # Pull calculated G(r) from cpdf
    ycalc = cpdf.evaluate()
    ydiff = yobs - ycalc

    rv = []
    rv += ax.plot(
        x, yobs, 'o', label='Gobs',
        markeredgecolor='blue', markerfacecolor='none'
    )
    rv += ax.plot(x, ycalc, color='red', label='Gcalc')
    rv += ax.plot(x, ydiff + baseline, label='Gdiff', color='green')
    rv += ax.plot(x, baseline + 0 * x, linestyle=':', color='black')

    ax.set_xlabel(u'r (Å)')
    ax.set_ylabel(u'G (Å$^{-2}$)')

    # Build DataFrame
    df = pd.DataFrame({'x': x, 'yobs': yobs, 'ycalc': ycalc, 'ydiff': ydiff})
    return rv, df

#=============================================================================
#=============================================================================
def visualize_fit_summary(cpdf, phase, output_plot_path,
                          font_size=14, label_font_size=20):
    """
    Visualize and summarize the fit results, including PDF data,
    bond lengths, and bond angles.

    Parameters:
    - fit: FitRecipe object (not used for evaluation here).
    - cpdf: PDFContribution object used in the refinement.
    - phase: Structure object for the current phase.
    - output_plot_path: String, directory path where output files will be saved.
    - font_size: Integer, font size for plot text (default: 14).
    - label_font_size: Integer, font size for axis labels and titles (default: 20).

    Returns:
    - None. Saves summary plots (PDF fit, bond‐length histograms, angle histograms).
    """

    import matplotlib.pyplot as plt

    def find_angle_triplets_full(bond_vectors):
        """
        Identify and calculate all center‐vertex‐vertex and vertex‐center‐center angles
        based entirely on detailed_composition.
        """
        angle_triplets = []
        unique_triplets = set()

        # dynamically pull centers & vertices
        centers  = [el for el,info in detailed_composition.items()
                    if info.get('polyhedron_center', False)]
        vertices = [el for el,info in detailed_composition.items()
                    if info.get('polyhedron_vertex', False)]

        # --- angles at each center: vertex-center-vertex ---
        for c in centers:
            # collect all bonds for this center
            bonds = []
            for v in vertices:
                bonds.extend(bond_vectors.get(f"{c}-{v}", []))
            # group bonds by central_atom index
            groups = {}
            for b in bonds:
                idx = b['central_atom']['index']
                groups.setdefault(idx, []).append(b)
            # for each central atom, make all unique angle pairs
            for idx, blist in groups.items():
                if len(blist) < 2:
                    continue
                for i in range(len(blist)):
                    for j in range(i+1, len(blist)):
                        b1, b2 = blist[i], blist[j]
                        lbl1 = b1['atom2']['label']
                        lbl2 = b2['atom2']['label']
                        center_lbl = b1['central_atom']['label']
                        key = tuple(sorted([center_lbl, lbl1, lbl2]))
                        if key in unique_triplets:
                            continue
                        angle = calculate_angle(b1['vector'], b2['vector'])
                        category = (
                            f"{b1['atom2']['symbol']}-"
                            f"{b1['central_atom']['symbol']}-"
                            f"{b2['atom2']['symbol']}"
                        )
                        angle_triplets.append({
                            'central_label':   center_lbl,
                            'atom1_label':     lbl1,
                            'atom2_label':     lbl2,
                            'angle':           angle,
                            'angle name':      (lbl1, center_lbl, lbl2),
                            'angle category':  category
                        })
                        unique_triplets.add(key)

        # --- angles at each vertex: center-vertex-center ---
        vdict = {}
        for c in centers:
            for v in vertices:
                for b in bond_vectors.get(f"{c}-{v}", []):
                    vidx = b['atom2']['index']
                    vdict.setdefault(vidx, []).append(b)
        for vidx, blist in vdict.items():
            if len(blist) < 2:
                continue
            for i in range(len(blist)):
                for j in range(i+1, len(blist)):
                    b1, b2 = blist[i], blist[j]
                    cent1 = b1['central_atom']['label']
                    cent2 = b2['central_atom']['label']
                    if cent1 == cent2:
                        continue
                    vert_lbl = b1['atom2']['label']
                    key = tuple(sorted([vert_lbl, cent1, cent2]))
                    if key in unique_triplets:
                        continue
                    angle = calculate_angle(b1['vector'], b2['vector'])
                    category = (
                        f"{b1['central_atom']['symbol']}-"
                        f"{b1['atom2']['symbol']}-"
                        f"{b2['central_atom']['symbol']}"
                    )
                    angle_triplets.append({
                        'central_label':   vert_lbl,
                        'atom1_label':     cent1,
                        'atom2_label':     cent2,
                        'angle':           angle,
                        'angle name':      (cent1, vert_lbl, cent2),
                        'angle category':  category
                    })
                    unique_triplets.add(key)

        return angle_triplets

    # 1) Calculate bond vectors (uses your global detailed_composition internally)
    bond_vectors   = get_polyhedral_bond_vectors(phase)
    angle_triplets = find_angle_triplets_full(bond_vectors)

    # 2) Set plot fonts
    plt.rcParams['font.size']   = font_size
    plt.rcParams['font.family'] = 'Arial'

    # 3) Create a 1×3 panel figure
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    # ---- Panel 1: PDF data vs. fit ----
    r      = cpdf.profile.x
    g_obs  = cpdf.profile.y
    g_calc = cpdf.evaluate()
    g_diff = g_obs - g_calc

    axs[0].plot(r, g_obs,  'o', label='G_obs',
                markeredgecolor='blue', markerfacecolor='none')
    axs[0].plot(r, g_calc,      label='G_calc')
    axs[0].plot(r, g_diff-4,    label='G_diff')
    axs[0].axhline(y=-4, linestyle=':', color='black')
    axs[0].set_xlabel('r (Å)',           fontsize=label_font_size)
    axs[0].set_ylabel('G (Å$^{-2}$)',    fontsize=label_font_size)
    axs[0].set_title('PDF Data and Fit', fontsize=label_font_size)
    axs[0].legend(fontsize=font_size)

    # ---- Panel 2: Bond‐length histograms ----
    for bond_type, blist in bond_vectors.items():
        left, right = bond_type.split('-')
        # skip pure vertex-vertex bonds
        if left in [el for el,info in detailed_composition.items() if info.get('polyhedron_vertex')] \
           and right in [el for el,info in detailed_composition.items() if info.get('polyhedron_vertex')]:
            continue
        lengths = [b['length'] for b in blist]
        sns.histplot(lengths, bins=50, kde=True, ax=axs[1], label=bond_type)
    axs[1].set_xlabel('Bond Length (Å)',           fontsize=label_font_size)
    axs[1].set_ylabel('Count',                     fontsize=label_font_size)
    axs[1].set_title('Bond Length Distributions',  fontsize=label_font_size)
    axs[1].legend(fontsize=font_size)

    # ---- Panel 3: Bond‐angle histograms ----
    categories = sorted({a['angle category'] for a in angle_triplets})
    for cat in categories:
        angles = [a['angle'] for a in angle_triplets if a['angle category']==cat]
        sns.histplot(angles,
                     bins=np.linspace(0, 180, 50), kde=True,
                     ax=axs[2], label=cat)
    axs[2].set_xlabel('Angle (°)',                  fontsize=label_font_size)
    axs[2].set_ylabel('Count',                      fontsize=label_font_size)
    axs[2].set_xlim(0, 180)
    axs[2].set_title('Bond Angle Distributions',    fontsize=label_font_size)
    axs[2].legend(fontsize=font_size)

    plt.tight_layout()
    plt.savefig(output_plot_path + 'summary_plot.pdf', dpi=600)
    plt.show()
#----------------------------------------------------------------------------    

def evaluate_and_plot(cpdf, fitting_range, csv_filename):
    """
    Evaluate model vs. data over a limited r-range, compute Rw,
    plot G_obs and G_sim with Rw annotation, and save to CSV.
    
    Parameters:
    - cpdf: PDFContribution object, with observed profile already set.
    - fitting_range: tuple or list [rmin, rmax]
    - csv_filename: str, path to output CSV file
    """
    import matplotlib.pyplot as plt
    # 1) Evaluate simulated G(r)
    g_sim = cpdf.evaluate()
    
    # 2) Extract observed data
    r = np.array(cpdf.profile.x)
    g_obs = np.array(cpdf.profile.y)
    
    # 3) Mask to fitting range
    rmin, rmax = fitting_range
    mask = (r >= rmin) & (r <= rmax)
    r_fit = r[mask]
    g_obs_fit = g_obs[mask]
    g_sim_fit = g_sim[mask]
    g_diff_fit = g_obs_fit - g_sim_fit
    
    # 4) Compute Rw
    #    Rw = sqrt( sum((Gobs-Gsim)^2) / sum(Gobs^2) )
    Rw = np.sqrt(np.sum(g_diff_fit**2) / np.sum(g_obs_fit**2))
    
    # 5) Plot
    plt.figure(figsize=(6,4))
    plt.plot(r_fit, g_obs_fit, 'o', label='G$_{obs}$', markersize=4, markeredgecolor='blue', markerfacecolor='none')
    plt.plot(r_fit, g_sim_fit,  '-', label='G$_{sim}$', linewidth=1.5, color='red')
    plt.xlabel('r (Å)')
    plt.ylabel('G(r) (Å$^{-2}$)')
    plt.title(f'PDF Simulation (Rw = {Rw:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 6) Save to CSV
    df = pd.DataFrame({
        'r':       r_fit,
        'G_obs':   g_obs_fit,
        'G_sim':   g_sim_fit,
        'G_diff':  g_diff_fit
    })
    df.to_csv(csv_filename, index=False)
    print(f"Saved data to {csv_filename}")
#----------------------------------------------------------------------------

def export_cifs(i, cpdf, output_path):
    """
    Export the refined structures from the PDFContribution object (`cpdf`) to CIF files.

    Parameters:
    - i: Integer, the step or iteration index used in naming the output files.
    - cpdf: PDFContribution object, containing the refined structural models for each phase.
    - output_path: String, the directory path where the exported CIF files will be saved.

    Returns:
    - None. The function writes CIF files to the specified `output_path`.
    """
    for phase in cpdf._generators:
        getattr(cpdf, str(phase)).stru.write(
            output_path + f"{phase}_{i}.cif", format='cif'
        )
    return


#=============================================================================
def saveResults(i, fit, cpdf, output_results_path):
    """
    Save the results of a refinement step, including fitting summary, visualizations,
    and data files.

    Parameters:
    - i: Integer, the step or iteration index used in naming the output files.
    - fit: FitRecipe object, containing the refined model and fit results.
    - cpdf: PDFContribution object, containing the same contribution that was used
            to build/refine `fit`.  This is needed for both plotmyfit() and export_cifs().
    - output_results_path: String, the directory path where results will be saved.

    Returns:
    - res: FitResults object, which contains a summary of the refinement results.
    """
    # Remove any constraint whose .par is None
    fit._oconstraints[:] = [c for c in fit._oconstraints if c.par is not None]

    # Create a FitResults summary and write it to disk
    res = FitResults(fit)
    res.saveResults(output_results_path + f'fitting_summary_{i}.txt')

    # Plot and save the “fit vs. data” figure, passing cpdf explicitly
    fig0, ax0 = subplots()
    fig, df = plotmyfit(cpdf, ax=ax0)
    fig0.savefig(output_results_path + f'fitting_{i}.png', dpi=600)

    # Save G(r), G_calc, and G_diff to CSV
    df.to_csv(output_results_path + f'fitting_curve_{i}.csv', index=False)

    # Export all phases to CIF files
    export_cifs(i, cpdf, output_results_path)

    # Capture fit.show() text output
    fit_show_filename = os.path.join(output_results_path, f"fit_state_{i}.txt")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fit.show()
    fit_show_text = buffer.getvalue()
    with open(fit_show_filename, "w") as f:
        f.write("==== Fit Recipe State (fit.show()) ====\n")
        f.write(fit_show_text)
        f.write("\n=======================================\n")

    return res


#-----------------------------------------------------------------------------
def finalize_results(cpdf, fit, output_results_path, myrange, myrstep):
    """
    Finalize the refinement process by exporting partial PDFs for each phase
    and generating a full-range extrapolated fit plot.

    Parameters:
    - cpdf: PDFContribution object, containing the fit results and phase contributions.
    - fit: FitRecipe object, the current fitting recipe used in the refinement.
    - output_results_path: String, path where the results will be saved.
    - myrange: Tuple, the range of r values for the extrapolated fit (e.g., (0.0, 80.0)).
    - myrstep: Float, the step size for r values.

    Returns:
    - None. Saves partial PDFs and the full-range extrapolated fit plot to the output directory.
    """
    # Export partial PDFs of each phase
    if len(cpdf._generators) > 1:
        scale_factors = {}

        # Read out the scale factors
        for phase in cpdf._generators:
            scale_factors[phase] = getattr(cpdf, 's_' + phase).value

        # Zero out the scale factors
        for phase in cpdf._generators:
            getattr(cpdf, 's_' + phase).value = 0

        # Generate a partial PDF for each phase
        for phase in scale_factors:
            getattr(cpdf, 's_' + phase).value = scale_factors[phase]
            fig0, ax0 = subplots()
            fig, df = plotmyfit(cpdf, ax=ax0)
            fig0.savefig(output_results_path + f'{phase}_partial_fit.png', dpi=600)
            df.to_csv(output_results_path + f'{phase}_partial.csv', index=False)
            getattr(cpdf, 's_' + phase).value = 0

        # Restore the original scale factors
        for phase in scale_factors:
            getattr(cpdf, 's_' + phase).value = scale_factors[phase]

    # Extrapolation of the fit to a full range regardless of the fitting steps
    cpdf.setCalculationRange(myrange[0], myrange[1], myrstep)
    fig0, ax0 = subplots()
    fig, df = plotmyfit(cpdf, ax=ax0)
    fig0.savefig(output_results_path + '_final_extrapolated_fit.png', dpi=600)
    df.to_csv(output_results_path + '_final_extrapolated_fit.csv', index=False)

    print(f"Results finalized and saved to {output_results_path}")


#=============================================================================
# PDF generation and handling
#=============================================================================
def simulatePDF(file, **config):
    """
    Simulate a single‐phase PDF, optionally damping with a spherical particle envelope.

    Parameters
    ----------
    file : str
        Path to a CIF (or other DiffPy‐readable) structure file.
    **config :
        rmin, rmax, rstep, qmin, qmax, qdamp, qbroad   ← for PDFCalculator
        psize                                        ← spherical particle diameter (Å)

    Returns
    -------
    df : pd.DataFrame
        Two‐column DataFrame with 'r' and 'g'.
    """
    # 1) split out the core PDF‐calculator args
    core = {k: v for k, v in config.items()
            if k in ('rmin','rmax','rstep','qmin','qmax','qdamp','qbroad')}
    # 2) make the calculator
    pdfc = PDFCalculator(**core)

    # 3) if the user passed psize=…, set the spherical‐shape envelope
    if 'psize' in config:
        pdfc.spdiameter = config['psize']

    # 4) load the structure and run
    stru = loadStructure(file)
    r, g = pdfc(stru)

    # 5) wrap in a DataFrame
    return pd.DataFrame({'r': r, 'g': g})



#----------------------------------------------------------------------------
def generatePDF(mypowderdata, composition, qmin=0.0, qmax=22, myrange=(1, 100), myrstep=0.01):
    """
    Generate a PDF from diffraction data with specified settings.

    Parameters:
    - mypowderdata: String, path to the powder diffraction data file.
    - composition: String, atomic composition of the sample (e.g., "O7 V2 Zr1").
    - qmin: Float, minimum Q value for the PDF generation (default: 0.0).
    - qmax: Float, maximum Q value for the PDF generation (default: 22).
    - myrange: Tuple of (rmin, rmax) for the PDF range (default: (1, 100)).
    - myrstep: Float, step size for r values (default: 0.01).

    Returns:
    - r0: Numpy array, r values of the PDF.
    - g0: Numpy array, g values of the PDF.
    - cfg: PDFConfig object, configuration used for the PDF generation.
    """
    cfg = PDFConfig(mode='xray',
                    composition=composition,
                    dataformat='QA',
                    rpoly=1.3,
                    rstep=myrstep,
                    rmin=myrange[0],
                    rmax=myrange[1])
    cfg.qmax = cfg.qmaxinst = qmax
    cfg.qmin = qmin
    pg0 = PDFGetter(config=cfg)
    r0, g0 = pg0(filename=mypowderdata)

    return r0, g0, cfg


#---------------------------------------------------------------------
def contribution(name, qmin, qmax, ciffile, periodic, super_cell):
    """
    Create and configure a PDFGenerator object from a CIF file.

    This function:
      - Loads a crystal structure from a specified CIF file
      - Optionally expands the structure to a given supercell
      - Sets the Q-range for PDF generation
      - Returns a PDFGenerator that can be used in a PDFContribution or FitRecipe

    Parameters:
    - name (str): A name to assign to the PDFGenerator instance.
    - qmin (float): Minimum Q value for PDF generation.
    - qmax (float): Maximum Q value for PDF generation.
    - ciffile (str): Path to the CIF file containing the crystal structure.
    - periodic (bool): Whether the structure is treated as periodic.
    - super_cell (tuple of int): Dimensions (nx, ny, nz) of the supercell used
      to replicate the structure, e.g. (1, 1, 1).

    Returns:
    - pdfgenerator (diffpy.srfit.pdf.pdfgenerator.PDFGenerator): The configured
      PDFGenerator object with the specified structure and Q-range.
    """
    global ncpu, pool
    pdfgenerator = PDFGenerator(str(name))
    pdfgenerator.setQmax(qmax)
    pdfgenerator.setQmin(qmin)
    pdfgenerator._calc.evaluatortype = 'OPTIMIZED'
    structure = loadStructure(ciffile)

    # allow an expansion to a supercell, which is defined by a 3-element tuple
    structure = expansion.supercell_mod.supercell(structure, (super_cell[0], super_cell[1], super_cell[2]))
    pdfgenerator.setStructure(structure, periodic=periodic)
    return pdfgenerator


#----------------------------------------------------------------------------
def DebyeContribution(name, qmin, qmax, ciffile, periodic, super_cell):
    """
    Create and configure a DebyePDFGenerator object from a CIF file,
    analogous to the original 'contribution' function but using the
    Debye real-space summation approach.

    Parameters:
    - name (str): A name/label for the DebyePDFGenerator.
    - qmin (float): Minimum Q value for the PDF calculation.
    - qmax (float): Maximum Q value for the PDF calculation.
    - ciffile (str): Path to the CIF file containing the structure.
    - periodic (bool): Whether to treat the structure as periodic
      (True) or as a finite cluster/molecule (False).
    - super_cell (tuple of int): The (nx, ny, nz) supercell dimensions
      used to expand the original unit cell.

    Returns:
    - pdfgenerator (DebyePDFGenerator): An object ready to be added
      to a PDFContribution or FitRecipe, just like the original function.
    """
    global ncpu, pool
    pdfgenerator = DebyePDFGenerator(str(name))
    pdfgenerator.setQmax(qmax)
    pdfgenerator.setQmin(qmin)
    pdfgenerator._calc.evaluatortype = 'OPTIMIZED'
    structure = loadStructure(ciffile)

    # allow an expansion to a supercell, which is defined by a 3-element tuple
    structure = expansion.supercell_mod.supercell(structure, (super_cell[0], super_cell[1], super_cell[2]))
    pdfgenerator.setStructure(structure, periodic=periodic)
    return pdfgenerator


#----------------------------------------------------------------------------
def phase(r0, g0, cfg, cif_directory, ciffile, fitRange, dx, qdamp, qbroad, name='cpdf'):
    """
    Create a PDFContribution object for the refinement process by combining experimental data
    with structure models from CIF files.

    Parameters:
    - r0: Numpy array, the observed r values of the PDF data.
    - g0: Numpy array, the observed g values of the PDF data.
    - cfg: PDFConfig object, containing the configuration for PDF generation.
    - cif_directory: String, directory path where the CIF files are stored.
    - ciffile: Dictionary, where keys are CIF filenames and values are lists containing
      [space group, periodicity (True/False), supercell dimensions (tuple)].
    - fitRange: Tuple of two floats, the r range for the fitting (xmin, xmax).
    - dx: Float, step size for r values in the fitting process.
    - qdamp: Float, damping parameter for the PDF (instrumental broadening).
    - qbroad: Float, broadening parameter for the PDF (sample-induced broadening).

    Returns:
    - cpdf: PDFContribution object, representing the combined phases with associated constraints
      and experimental data.
    """
    global ncpu, pool
    
    cpdf = PDFContribution(name)
    cpdf.setScatteringType('X')
    cpdf.setQmax(cfg.qmax)
    cpdf.profile.setObservedProfile(r0.copy(), g0.copy())
    cpdf.setCalculationRange(xmin=fitRange[0], xmax=fitRange[1], dx=dx)

    # specify an independent structure model
    for i, file in enumerate(ciffile.keys()):
        print('Phase: ', i, file)
        periodic = list(ciffile.values())[i][1]
        super_cell = list(ciffile.values())[i][2]
        print('Phase periodic? ', periodic)
        pdf = contribution('Phase' + str(i), cfg.qmin, cfg.qmax, cif_directory + file, periodic, super_cell)
        pdf.qdamp.value = qdamp
        pdf.qbroad.value = qbroad
        pdf.parallel(ncpu=ncpu, mapfunc=pool.map)
        cpdf.addProfileGenerator(pdf)

    cpdf.setResidualEquation('resv')
    return cpdf


# =============================================================================
# Rigig body and aconnectivity
# =============================================================================

def lattice_vectors(a, b, c, alpha, beta, gamma):
    """
    Calculate the lattice transformation matrix from lattice parameters.

    Parameters:
    - a: Float, lattice parameter a in Å.
    - b: Float, lattice parameter b in Å.
    - c: Float, lattice parameter c in Å.
    - alpha: Float, lattice angle α in degrees (between b and c).
    - beta: Float, lattice angle β in degrees (between a and c).
    - gamma: Float, lattice angle γ in degrees (between a and b).

    Returns:
    - lattice_matrix: 3x3 numpy array, with columns representing lattice vectors 
      in Cartesian coordinates. Can be used to convert fractional coordinates to Cartesian.
    """
    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    v_x = [a, 0, 0]
    v_y = [b * np.cos(gamma_r), b * np.sin(gamma_r), 0]
    c_x = c * np.cos(beta_r)
    c_y = c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
    c_z = np.sqrt(c**2 - c_x**2 - c_y**2)
    v_z = [c_x, c_y, c_z]
    return np.array([v_x, v_y, v_z]).T  # shape (3, 3)


#----------------------------------------------------------------------------
def calculate_angle(v1, v2):
    """
    Calculate the angle in degrees between two vectors in 3D space.

    Parameters:
    - v1: Numpy array or list, the first vector (e.g., [x1, y1, z1]).
    - v2: Numpy array or list, the second vector (e.g., [x2, y2, z2]).

    Returns:
    - angle: Float, the angle between `v1` and `v2` in degrees. The result is in the range [0, 180].
    """
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norm_product
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return angle


#----------------------------------------------------------------------------
def calculate_dihedral(v1, v2, v3):
    """
    Calculate the dihedral angle in degrees between planes defined by v1, v2, and v3.

    Parameters:
    - v1: Vector from atom A1 to A2.
    - v2: Vector from atom A2 to A3.
    - v3: Vector from atom A3 to A4.

    Returns:
    - dihedral_angle: Float, the dihedral angle in degrees. Returns None if vectors are invalid.
    """
    import numpy as _np

    # Validate non-zero vectors
    def is_zero_vector(vec):
        return _np.linalg.norm(vec) < 1e-12

    if is_zero_vector(v1) or is_zero_vector(v2) or is_zero_vector(v3):
        print(f"Zero vector detected: v1={v1}, v2={v2}, v3={v3}")
        return None

    # Normalize the vectors
    v1 = v1 / _np.linalg.norm(v1)
    v2 = v2 / _np.linalg.norm(v2)
    v3 = v3 / _np.linalg.norm(v3)

    # Calculate the normals to the planes
    n1 = _np.cross(v1, v2)
    n2 = _np.cross(v2, v3)

    # Validate non-zero normals
    if is_zero_vector(n1) or is_zero_vector(n2):
        print(f"Zero normal vector detected: n1={n1}, n2={n2}")
        return None

    n1 = n1 / _np.linalg.norm(n1)
    n2 = n2 / _np.linalg.norm(n2)

    # Calculate dihedral angle
    x = _np.dot(n1, n2)
    y = _np.dot(v2, _np.cross(n1, n2))

    # Handle numerical precision issues with arccos and arctan2
    x = _np.clip(x, -1.0, 1.0)
    dihedral_angle = np.degrees(np.arctan2(y, x))

    return dihedral_angle


#----------------------------------------------------------------------------
def get_polyhedral_bond_vectors(phase, max_workers=4):
    """
    Compute bond vectors for every center–vertex pair (e.g. Zr–O, V–O, …)
    and all vertex–vertex bonds (e.g. O–O, or any other vertex–vertex)
    driven entirely by the global detailed_composition dict.

    Returns:
      bond_vectors: dict with keys "<Center>-<Vertex>" and "<Vertex1>-<Vertex2>",
      each mapping to a list of bond‐info dicts:
        • For center–vertex bonds:
          {
            'central_atom':   {symbol,index,label,position_frac},
            'atom1':          {symbol,index,label,position_frac},  # same as central_atom
            'atom2':          {symbol,index,label,position_frac},  # the vertex
            'vector':         cartesian bond vector,
            'length':         cartesian bond length,
            'relative_length':fractional‐coord bond length
          }
        • For vertex–vertex bonds:
          {
            'central_atom':   {symbol,index,label,position_frac},  # the polyhedron center
            'atom1':          {symbol,index,label,position_frac},  # first vertex
            'atom2':          {symbol,index,label,position_frac},  # second vertex
            'vector':         cartesian edge vector,
            'length':         cartesian edge length,
            'relative_length':fractional‐coord edge length
          }
    """
    from itertools import combinations_with_replacement, combinations

    # 1) Decide which elements are centers vs. vertices
    centers  = [el for el,info in detailed_composition.items() if info.get('polyhedron_center', False)]
    vertices = [el for el,info in detailed_composition.items() if info.get('polyhedron_vertex', False)]

    # 2) Initialize empty lists for each center–vertex and each vertex–vertex type
    bond_vectors = {}
    for c in centers:
        for v in vertices:
            bond_vectors[f"{c}-{v}"] = []
    for v1, v2 in combinations_with_replacement(vertices, 2):
        bond_vectors[f"{v1}-{v2}"] = []

    # 3) Build lattice matrix and position maps
    lat = phase.lattice
    a, b, c = lat.a.value, lat.b.value, lat.c.value
    lattice_mat = lattice_vectors(a, b, c,
                                  lat.alpha.value,
                                  lat.beta.value,
                                  lat.gamma.value)
    scatterers = phase.getScatterers()
    pos_frac = {i: np.array([atm.x.value, atm.y.value, atm.z.value])
                for i, atm in enumerate(scatterers)}
    pos_cart = {i: lattice_mat.dot(frac) for i, frac in pos_frac.items()}
    elements = {i: atm.element for i, atm in enumerate(scatterers)}
    labels   = {i: atm.name.upper() for i, atm in enumerate(scatterers)}

    # 4) Gather which vertices coordinate each center
    coord = {i: [] for i, el in elements.items() if el in centers}

    # 5) Center→vertex bonds using per‐center cutoffs
    for i, el in elements.items():
        if el in centers:
            cpos   = pos_cart[i]
            cutoff = detailed_composition[el]['cutoff']
            for j, elj in elements.items():
                if elj in vertices and i != j:
                    vec    = pos_cart[j] - cpos
                    length = np.linalg.norm(vec)
                    if cutoff[0] <= length <= cutoff[1]:
                        rel_len = np.linalg.norm(pos_frac[j] - pos_frac[i])
                        info = {
                            'central_atom': {
                                'symbol': el, 'index': i, 'label': labels[i],
                                'position': pos_frac[i]
                            },
                            'atom1': {
                                'symbol': el, 'index': i, 'label': labels[i],
                                'position': pos_frac[i]
                            },
                            'atom2': {
                                'symbol': elj,'index': j, 'label': labels[j],
                                'position': pos_frac[j]
                            },
                            'vector':         vec,
                            'length':         length,
                            'relative_length':rel_len
                        }
                        bond_vectors[f"{el}-{elj}"].append(info)
                        coord[i].append(j)

    # 6) Vertex–vertex bonds within each polyhedron
    for cen_idx, verts in coord.items():
        cinfo = {
            'symbol':   elements[cen_idx],
            'index':    cen_idx,
            'label':    labels[cen_idx],
            'position': pos_frac[cen_idx]
        }
        for u, v in combinations(verts, 2):
            vec     = pos_cart[v] - pos_cart[u]
            length  = np.linalg.norm(vec)
            rel_len = np.linalg.norm(pos_frac[v] - pos_frac[u])
            el_u, el_v = elements[u], elements[v]
            key = f"{min(el_u,el_v)}-{max(el_u,el_v)}"
            edge_info = {
                'central_atom':   cinfo,
                'atom1': {
                    'symbol': el_u, 'index': u, 'label': labels[u],
                    'position': pos_frac[u]
                },
                'atom2': {
                    'symbol': el_v, 'index': v, 'label': labels[v],
                    'position': pos_frac[v]
                },
                'vector':         vec,
                'length':         length,
                'relative_length':rel_len
            }
            bond_vectors[key].append(edge_info)

    return bond_vectors





#----------------------------------------------------------------------------
def find_bond_pairs(phase, bond_vectors):
    """
    Identify bond pairs from the bond_vectors data and filter them based on phase-specific parameters.
    Leverage atom positions provided in the bond_vectors.

    Parameters:
    - phase: The current phase of the structure being analyzed.
    - bond_vectors: Dictionary containing bond information for 'Zr-O', 'V-O', and 'O-O' bond types.

    Returns:
    - bond_pairs: A list of dictionaries representing each bond pair that meets phase-specific criteria.
    """
    bond_pairs = []
    phase_added_params = added_params[str(phase)]

    for bond_type, bonds in bond_vectors.items():
        for bond in bonds:
            atom1_label = bond['atom1']['label']
            atom2_label = bond['atom2']['label']

            if atom1_label in phase_added_params and atom2_label in phase_added_params:
                bond_pair = {
                    'bond_type': bond_type,
                    'atom1_label': atom1_label,
                    'atom2_label': atom2_label,
                    'atom1_position': bond['atom1']['position'],
                    'atom2_position': bond['atom2']['position'],
                    'central_label': bond['central_atom']['label'],
                    'central_position': bond['central_atom']['position'],
                    'length': bond['length'],
                    'relative_length': bond['relative_length'],
                    'vector': bond['vector']
                }
                bond_pairs.append(bond_pair)

    return bond_pairs


#----------------------------------------------------------------------------

def find_angle_triplets(bond_pairs, include_range=(30, 175)):
    from itertools import combinations
    """
    Identify unique angle triplets around each central atom, including and excluding the central atom,
    while ensuring only unique triplets are calculated. Includes only O-O-O angles in the specified range.

    Parameters:
    - bond_pairs: List of dictionaries representing each bond pair with bond type and atom details.
    - include_range: Tuple specifying the range of angles to include (min, max).

    Returns:
    - angle_triplets: A list of dictionaries, each representing a unique angle triplet.
    """
    angle_triplets = []
    unique_triplets = set()

    bonds_by_central_atom = {}
    for bond in bond_pairs:
        central_label = bond['central_label']
        bonds_by_central_atom.setdefault(central_label, []).append(bond)

    for central_label, bonds in bonds_by_central_atom.items():
        neighbor_atoms = []
        for bond in bonds:
            if bond['atom1_label'] != central_label:
                neighbor_atoms.append((bond['atom1_label'], bond['vector']))
            else:
                neighbor_atoms.append((bond['atom2_label'], bond['vector']))

        for atom_combo in combinations(neighbor_atoms, 2):
            atom1_label, vector1 = atom_combo[0]
            atom2_label, vector2 = atom_combo[1]
            triplet_sorted = tuple(sorted([atom1_label, central_label, atom2_label]))
            if triplet_sorted not in unique_triplets and atom1_label != atom2_label:
                angle = calculate_angle(vector1, vector2)
                angle_triplets.append({
                    'central_label': central_label,
                    'atom1_label': atom1_label,
                    'atom2_label': atom2_label,
                    'angle': angle,
                    'angle name': (atom1_label, central_label, atom2_label),
                    'angle category': f"{atom1_label}-{central_label}-{atom2_label}"
                })
                unique_triplets.add(triplet_sorted)

        for atom_combo in combinations(neighbor_atoms, 3):
            atom1_label, vector1 = atom_combo[0]
            atom2_label, vector2 = atom_combo[1]
            atom3_label, vector3 = atom_combo[2]
            triplet_sorted = tuple(sorted([atom1_label, atom2_label, atom3_label]))
            if triplet_sorted not in unique_triplets and len({atom1_label, atom2_label, atom3_label}) == 3:
                if all("O" in atom for atom in (atom1_label, atom2_label, atom3_label)):
                    angle1 = calculate_angle(vector1 - vector2, vector3 - vector2)
                    if include_range[0] <= angle1 <= include_range[1]:
                        angle_triplets.append({
                            'central_label': central_label,
                            'atom1_label': atom1_label,
                            'atom2_label': atom2_label,
                            'atom3_label': atom3_label,
                            'angle': angle1,
                            'angle name': (atom1_label, atom2_label, atom3_label),
                            'angle category': f"{atom1_label}-{atom2_label}-{atom3_label}"
                        })
                        unique_triplets.add(triplet_sorted)
                else:
                    angle1 = calculate_angle(vector1 - vector2, vector3 - vector2)
                    angle_triplets.append({
                        'central_label': central_label,
                        'atom1_label': atom1_label,
                        'atom2_label': atom2_label,
                        'atom3_label': atom3_label,
                        'angle': angle1,
                        'angle name': (atom1_label, atom2_label, atom3_label),
                        'angle category': f"{atom1_label}-{atom2_label}-{atom3_label}"
                    })
                    unique_triplets.add(triplet_sorted)

    return angle_triplets


#----------------------------------------------------------------------------
def find_dihedral_quadruplets(bond_pairs, angle_threshold=2.0):
    """
    Identify and calculate dihedral quadruplets from bond pairs while ensuring they belong to the same polyhedron.

    Parameters:
    - bond_pairs: List of dictionaries representing bond pairs.
    - angle_threshold: Float, minimum absolute dihedral angle to consider (default: 3 degrees).

    Returns:
    - dihedral_quadruplets: List of dictionaries, each representing a valid dihedral quadruplet with its angle.
    """
    dihedral_quadruplets = []
    unique_quadruplets = set()

    for bond1, bond2 in combinations(bond_pairs, 2):
        if bond1['central_label'] != bond2['central_label']:
            continue

        polyhedron_center = bond1['central_label']
        atoms = {bond1['atom1_label'], bond1['atom2_label'], bond2['atom1_label'], bond2['atom2_label']}
        if len(atoms) != 4:
            continue

        atom1, atom2, atom3, atom4 = sorted(list(atoms))
        pos1 = np.array(bond1['atom1_position'])
        pos2 = np.array(bond1['atom2_position'])
        pos3 = np.array(bond2['atom1_position'])
        pos4 = np.array(bond2['atom2_position'])

        unique_positions = {tuple(pos1), tuple(pos2), tuple(pos3), tuple(pos4)}
        if len(unique_positions) != 4:
            continue

        v1 = pos2 - pos1
        v2 = pos3 - pos2
        v3 = pos4 - pos3

        if (
            np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6 or np.linalg.norm(v3) < 1e-6 or
            np.linalg.norm(np.cross(v1, v2)) < 1e-12
        ):
            continue

        dihedral_angle = calculate_dihedral(v1, v2, v3)
        if dihedral_angle is None:
            continue

        dihedral_angle_degrees = dihedral_angle  # already in degrees
        if abs(dihedral_angle_degrees) < angle_threshold or abs(dihedral_angle_degrees) > (180 - angle_threshold):
            continue

        quadruplet = tuple(sorted([atom1, atom2, atom3, atom4]))
        if quadruplet in unique_quadruplets:
            continue

        dihedral_quadruplets.append({
            'quadruplet': quadruplet,
            'angle': dihedral_angle_degrees,
            'polyhedron_center': polyhedron_center
        })
        unique_quadruplets.add(quadruplet)

    return dihedral_quadruplets


#----------------------------------------------------------------------------
def detect_edge_bonds(bond_vectors, threshold=0.005):
    """
    Identify any center–vertex bonds where at least one atom lies within `threshold`
    of a periodic unit‐cell boundary (0 or 1 in fractional coords).

    Parameters:
    - bond_vectors: dict of lists returned by get_polyhedral_bond_vectors(),
      with keys like "Zr-O", "V-O", etc., and “vertex-vertex”.
    - threshold: float, distance from the 0/1 boundary (default: 0.005).

    Returns:
    - edge_bonds: dict mapping each center–vertex bond type to the sub‐list of bonds
      that lie near a cell edge.
    """
    def _near_edge(pos):
        # pos is a length-3 array of fractional coords
        return any(c < threshold or c > 1.0 - threshold for c in pos)

    edge_bonds = {}
    for bond_type, blist in bond_vectors.items():
        # skip polygon edges if you only care about center–vertex
        if bond_type == 'vertex-vertex':
            continue
        edge_bonds[bond_type] = []
        for bond in blist:
            cpos = bond['center' if 'center' in bond else 'atom1']['position']
            vpos = bond['vertex' if 'vertex' in bond else 'atom2']['position']
            if _near_edge(cpos) or _near_edge(vpos):
                edge_bonds[bond_type].append(bond)

    total = sum(len(v) for v in edge_bonds.values())
    print(f"[INFO] Detected {total} center–vertex bonds near any cell edge.")
    return edge_bonds



#----------------------------------------------------------------------------
def create_constrain_expressions_for_bonds(bond_pairs, phase):
    """
    Create constraint expressions for bond lengths between two atoms for a given phase.

    Parameters:
    - bond_pairs: List of dictionaries representing bond pairs with labels and bond lengths.
    - phase: The phase identifier.

    Returns:
    - constrain_dict: Dictionary containing constraint expressions for each bond.
    """
    constrain_dict = {}
    for bond in bond_pairs:
        atom1_label = bond['atom1_label']
        atom2_label = bond['atom2_label']
        relative_length = bond['relative_length']

        var_name = f'bond_length_{atom1_label}_{atom2_label}_{phase}'
        bond_expr = (
            f"((x_{atom1_label}_{phase} - x_{atom2_label}_{phase})**2 + "
            f"(y_{atom1_label}_{phase} - y_{atom2_label}_{phase})**2 + "
            f"(z_{atom1_label}_{phase} - z_{atom2_label}_{phase})**2)**0.5"
        )

        constrain_dict[var_name] = {
            'atoms': (atom1_label, atom2_label),
            'expression': bond_expr,
            'relative_length': relative_length
        }

    return constrain_dict


#----------------------------------------------------------------------------
def create_constrain_expressions_for_angles(angle_triplets, phase):
    """
    Generate constraint expressions for angles formed by atom triplets in a given phase.

    Parameters:
    - angle_triplets: List of dictionaries representing angle triplets with angle values and labels.
    - phase: String, the phase identifier.

    Returns:
    - constrain_dict: Dictionary where keys are variable names for angles, and values are
      dictionaries containing the constraint expression, angle value, and involved atoms.
    """
    constrain_dict = {}
    for angle in angle_triplets:
        angle_name = angle['angle name']
        angle_value = angle['angle']

        if angle_value is None or np.isnan(angle_value):
            continue

        var_name = f'angle_{"_".join(angle_name)}_{phase}'
        atom1_label, central_label, atom2_label = angle_name
        angle_expr = (
            f"arccos(((x_{atom1_label}_{phase} - x_{central_label}_{phase}) * (x_{atom2_label}_{phase} - x_{central_label}_{phase}) + "
            f"(y_{atom1_label}_{phase} - y_{central_label}_{phase}) * (y_{atom2_label}_{phase} - y_{central_label}_{phase}) + "
            f"(z_{atom1_label}_{phase} - z_{central_label}_{phase}) * (z_{atom2_label}_{phase} - z_{central_label}_{phase})) / "
            f"((sqrt((x_{atom1_label}_{phase} - x_{central_label}_{phase})**2 + "
            f"(y_{atom1_label}_{phase} - y_{central_label}_{phase})**2 + "
            f"(z_{atom1_label}_{phase} - z_{central_label}_{phase})**2)) * "
            f"(sqrt((x_{atom2_label}_{phase} - x_{central_label}_{phase})**2 + "
            f"(y_{atom2_label}_{phase} - y_{central_label}_{phase})**2 + "
            f"(z_{atom2_label}_{phase} - z_{central_label}_{phase})**2))))"
        )

        constrain_dict[var_name] = {
            'atoms': (atom1_label, central_label, atom2_label),
            'expression': angle_expr,
            'angle': angle_value
        }

    return constrain_dict


#----------------------------------------------------------------------------
def create_constrain_expressions_for_dihedrals(dihedral_quadruplets, phase):
    """
    Generate constraint expressions for dihedral angles formed by atom quadruplets in a given phase.

    Parameters:
    - dihedral_quadruplets: List of dictionaries representing dihedral quadruplets with angle values and labels.
    - phase: String, the phase identifier.

    Returns:
    - constrain_dict: Dictionary where keys are variable names for dihedral angles, and values are
      dictionaries containing the constraint expression, angle value, and involved atoms.
    """
    constrain_dict = {}

    for dihedral in dihedral_quadruplets:
        quadruplet = dihedral['quadruplet']
        dihedral_value = dihedral['angle']
        polyhedron_center = dihedral['polyhedron_center']

        if dihedral_value is None or np.isnan(dihedral_value):
            continue

        atom1, atom2, atom3, atom4 = quadruplet
        var_name = f"dihedral_{'_'.join(quadruplet)}_{phase}"
        dihedral_expr = (
            f"arctan(((x_{atom3}_{phase} - x_{atom2}_{phase}) * "
            f"(((y_{atom2}_{phase} - y_{atom1}_{phase}) * (z_{atom4}_{phase} - z_{atom3}_{phase})) - "
            f"((z_{atom2}_{phase} - z_{atom1}_{phase}) * (y_{atom4}_{phase} - y_{atom3}_{phase}))) + "
            f"((y_{atom3}_{phase} - y_{atom2}_{phase}) * "
            f"(((z_{atom2}_{phase} - z_{atom1}_{phase}) * (x_{atom4}_{phase} - x_{atom3}_{phase})) - "
            f"((x_{atom2}_{phase} - x_{atom1}_{phase}) * (z_{atom4}_{phase} - z_{atom3}_{phase}))) + "
            f"((z_{atom3}_{phase} - z_{atom2}_{phase}) * "
            f"(((x_{atom2}_{phase} - x_{atom1}_{phase}) * (y_{atom4}_{phase} - y_{atom3}_{phase})) - "
            f"((y_{atom2}_{phase} - y_{atom1}_{phase}) * (x_{atom4}_{phase} - x_{atom3}_{phase}))) )) / "
            f"(((x_{atom2}_{phase} - x_{atom1}_{phase}) * (x_{atom3}_{phase} - x_{atom2}_{phase})) + "
            f"((y_{atom2}_{phase} - y_{atom1}_{phase}) * (y_{atom3}_{phase} - y_{atom2}_{phase})) + "
            f"((z_{atom2}_{phase} - z_{atom1}_{phase}) * (z_{atom3}_{phase} - z_{atom2}_{phase})) )))"
        )

        constrain_dict[var_name] = {
            'atoms': (atom1, atom2, atom3, atom4),
            'expression': dihedral_expr,
            'angle': dihedral_value,
            'polyhedron center': polyhedron_center
        }

    return constrain_dict


#----------------------------------------------------------------------------
def refinement_RigidBody(fit, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False):
    """
    Extend the fitting recipe with rigid body restraints based on bonds, angles, and dihedrals.

    Parameters:
    - fit: FitRecipe object, containing the fitting recipe.
    - cpdf: PDFContribution object, representing the PDF model.
    - constrain_bonds: Tuple (Boolean, Float), enable bond constraints and specify the sigma for bond restraint.
    - constrain_angles: Tuple (Boolean, Float), enable angle constraints and specify the sigma for angle restraint.
    - constrain_dihedrals: Tuple (Boolean, Float), enable dihedral constraints.
    - adaptive: Boolean, if True, recalculate bond vectors dynamically based on the current structure.

    Returns:
    - fit: Updated FitRecipe object with added constraints.
    """

    print("\n--------------------------------------")
    print("[INFO] Adding constraints on bonds, angles, and dihedrals")
    print("--------------------------------------")

    global global_bond_vectors

    for i, phase in enumerate(cpdf._generators):
        phase_key = str(phase)

        # Retrieve or recalculate bond vectors
        if not adaptive:
            bond_vectors = global_bond_vectors[phase_key]
        else:
            bond_vectors = get_polyhedral_bond_vectors(getattr(cpdf, phase).phase)

        # Detect edge bonds near cell boundaries
        edge_bonds = detect_edge_bonds(bond_vectors, threshold=0.005)

        # Find bond pairs, angles, and dihedral triplets
        bond_pairs = find_bond_pairs(phase_key, bond_vectors)
        angle_triplets = find_angle_triplets(bond_pairs)
        dihedral_quadruplets = find_dihedral_quadruplets(bond_pairs)

        # Generate constraint expressions
        constrain_dict_bonds = create_constrain_expressions_for_bonds(bond_pairs, phase_key)
        constrain_dict_angles = create_constrain_expressions_for_angles(angle_triplets, phase_key)
        constrain_dict_dihedrals = create_constrain_expressions_for_dihedrals(dihedral_quadruplets, phase_key)

        # --------------------------------------------------------------------
        # BOND CONSTRAINTS
        # --------------------------------------------------------------------
        if constrain_bonds[0]:
            print("\n[INFO] Processing bond constraints...")

            # Step 1: Identify shared vertex atoms for center-vertex-center bonds
            shared_vertex_dict = {}
            centers  = [el for el,info in detailed_composition.items() if info['polyhedron_center']]
            vertices = [el for el,info in detailed_composition.items() if info['polyhedron_vertex']]

            for var_name, data in constrain_dict_bonds.items():
                atom1_label, atom2_label = data['atoms']
                # figure out which label is center vs. vertex
                center_lbl = None
                vertex_lbl = None
                if any(atom1_label.startswith(c) for c in centers) and any(atom2_label.startswith(v) for v in vertices):
                    center_lbl, vertex_lbl = atom1_label, atom2_label
                elif any(atom2_label.startswith(c) for c in centers) and any(atom1_label.startswith(v) for v in vertices):
                    center_lbl, vertex_lbl = atom2_label, atom1_label
                if vertex_lbl:
                    shared_vertex_dict.setdefault(vertex_lbl, {'bond_vars': [], 'expressions': []})
                    shared_vertex_dict[vertex_lbl]['bond_vars'].append(var_name)
                    shared_vertex_dict[vertex_lbl]['expressions'].append(data['expression'])

            # keep only those vertices bonded to two centers
            shared_vertex_dict = {
                v:info for v,info in shared_vertex_dict.items()
                if len(info['bond_vars']) == 2
            }

            # Step 2: Identify center-vertex bonds at the unit cell edge
            edge_bond_dict = {}
            for btype, blist in edge_bonds.items():
                for bond in blist:
                    atom1_label = bond["atom1"]["label"]
                    atom2_label = bond["atom2"]["label"]
                    bond_var_name = f"bond_length_{atom1_label}_{atom2_label}_{phase_key}"
                    edge_bond_dict.setdefault(atom2_label, []).append(bond_var_name)

            print(f"[INFO] Identified {sum(len(lst) for lst in edge_bond_dict.values())} edge bonds.")

            # Step 3: Identify problematic bonds (outside cutoff ranges)
            bond_vectors_current = get_polyhedral_bond_vectors(getattr(cpdf, phase).phase)
            problematic_bonds_dict = {}
            for bond_type, blist in bond_vectors_current.items():
                if bond_type == 'O-O':
                    continue
                center_sym = bond_type.split('-')[0]
                cutoff = detailed_composition[center_sym]['cutoff']
                for bond in blist:
                    length = bond['length']
                    if length < cutoff[0] or length > cutoff[1]:
                        atom1_label = bond["atom1"]["label"]
                        atom2_label = bond["atom2"]["label"]
                        bond_key = f"bond_length_{atom1_label}_{atom2_label}_{phase_key}"
                        problematic_bonds_dict[bond_key] = bond

            print(f"[INFO] Detected {len(problematic_bonds_dict)} problematic bonds (outside cutoff).")

            # Step 4: Apply constraints
            for var_name, data in constrain_dict_bonds.items():
                try:
                    fit.newVar(var_name, tags=['bond_length'])
                    fit.constrain(var_name, data['expression'])

                    # decide sigma & bounds based on category
                    if any(var_name in info['bond_vars'] for info in shared_vertex_dict.values()):
                        category = "SHARED-VERTEX BOND"
                        sigma = 1e-7
                    elif any(var_name in bond_list for bond_list in edge_bond_dict.values()):
                        category = "EDGE BOND"
                        sigma = 1e-7
                    elif var_name in problematic_bonds_dict:
                        category = "PROBLEMATIC BOND"
                        sigma = 1e-7
                    else:
                        category = "NORMAL BOND"
                        sigma = constrain_bonds[1]

                    rel_len = data['relative_length']
                    lb, ub = 0.95 * rel_len, 1.05 * rel_len
                    print(f"[{category}] {var_name}: lb={lb:.6f}, ub={ub:.6f}, sigma={sigma}")

                    fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=sigma)

                except Exception as e:
                    print(f"[WARNING] Skipped {var_name} due to error: {e}")

        # --------------------------------------------------------------------
        # ANGLE CONSTRAINTS
        # --------------------------------------------------------------------
        if constrain_angles[0]:
            print("\n[INFO] Processing angle constraints...")
            for var_name, data in constrain_dict_angles.items():
                try:
                    ang_limit = 1.0  # in degrees
                    fit.newVar(var_name, tags=['bond_angle'])
                    fit.constrain(var_name, data['expression'])
                    lb = np.radians(data['angle'] - ang_limit)
                    ub = np.radians(data['angle'] + ang_limit)
                    fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=constrain_angles[1])

                    print(f"[INFO] {var_name}: angle={data['angle']:.2f}° (±{ang_limit}°)")
                except Exception as e:
                    print(f"[WARNING] Skipped angle {var_name} due to error: {e}")

        # --------------------------------------------------------------------
        # DIHEDRAL CONSTRAINTS
        # --------------------------------------------------------------------
        if constrain_dihedrals[0]:
            print("\n[INFO] Processing dihedral angle constraints...")
            for var_name, data in constrain_dict_dihedrals.items():
                try:
                    ang_limit = 2.0  # in degrees
                    fit.newVar(var_name, tags=['dihedral_angle'])
                    fit.constrain(var_name, data['expression'])
                    lb = np.radians(data['angle'] - ang_limit)
                    ub = np.radians(data['angle'] + ang_limit)
                    fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=constrain_dihedrals[1])

                    print(f"[INFO] {var_name}: dihedral={data['angle']:.2f}° (±{ang_limit}°)")
                except Exception as e:
                    print(f"[WARNING] Skipped dihedral {var_name} due to error: {e}")

        # --------------------------------------------------------------------
        # Cleanup: Remove variables that are None
        # --------------------------------------------------------------------
        for name in dir(fit):
            if name.startswith(('bond_length_', 'angle_', 'dihedral_')):
                try:
                    var = getattr(fit, name)
                    if var.value is None:
                        fit.unconstrain(var)
                        fit.delVar(var)
                        print(f"[INFO] Removed '{name}' due to a missing or incorrect value.")
                except Exception:
                    continue

    return fit



# =============================================================================
# Helper functions
# =============================================================================
def map_sgpar_params(sgpar, attr_name):
    """
    Dynamically map a list of parameters (e.g. xyzpars or adppars)
    to their corresponding atoms, using the specified attribute name.

    Parameters:
    sgpar (obj): Space group parameter object returned by constrainAsSpaceGroup.
    attr_name (str): The name of the attribute to map, e.g. 'xyzpars' or 'adppars'.

    Returns:
    dict: A mapping from parameter names to (mapped_label, atom_name_upper).
    """
    param_list = getattr(sgpar, attr_name)
    mapped_params = {}

    for par in param_list:
        parname = par.name
        parts = parname.split("_")
        coord = parts[0]
        idx = int(parts[1])
        scatterer = sgpar.scatterers[idx]
        atom_name = scatterer.name
        mapped_label = f"{coord}_{atom_name.upper()}"
        mapped_params[parname] = (mapped_label, atom_name.upper())

    return mapped_params


#----------------------------------------------------------------------------


# =============================================================================
# Recipe and fitting functions
# =============================================================================
def fit_me(i, fitting_range, myrstep, fitting_order, fit, cpdf, residualEquation, output_results_path, **convergence_options):
    """
    Perform the refinement process for a single fitting step.

    Parameters:
    - i: Integer, index of the fitting step.
    - fitting_range: List of two floats, the r range for the fit (e.g., [1.5, 27]).
    - myrstep: Float, step size for r values.
    - fitting_order: List of strings, tags defining the fitting order (e.g., ['lat', 'scale', 'xyz']).
    - fit: FitRecipe object, the current fitting recipe.
    - cpdf: PDFContribution object, representing the PDF model.
    - residualEquation: String, residual equation used for fitting (e.g., 'resv').
    - output_results_path: String, path to save the fitting results.
    - **convergence_options: Additional keyword arguments for the optimizer.

    Returns:
    - None
    """
    print('-------------------------------------------------------------------')
    print(f"Fitting stage {i}")
    print('-------------------------------------------------------------------')
    cpdf.setCalculationRange(fitting_range[0], fitting_range[1], myrstep)
    cpdf.setResidualEquation(residualEquation)
    fit.fix('all')
    for step in fitting_order:
        try:
            print(f'Freeing parameter: {step}')
            fit.free(step)
            optimizer = 'L-BFGS-B'
            #minimize(fit.scalarResidual, fit.values, method=optimizer, options=convergence_options)
        except Exception:
            continue

    res = saveResults(i, fit, cpdf, output_results_path)

    # Statistics on the bond lengths and angle distributions
    for phase_name in cpdf._generators:
        phase = getattr(cpdf, str(phase_name)).phase
        visualize_fit_summary(cpdf, phase, output_results_path + f"{i}_{phase_name}_", font_size=14, label_font_size=20)
    return


#-----------------------------------------------------------------------------
def refinement_basic(cpdf, anisotropic, unified_Uiso, sgoffset=[0, 0, 0]):
    """
    Create a basic fitting recipe with default constraints based on the space group.

    Parameters:
    - cpdf: PDFContribution object, representing the PDF model.
    - anisotropic: Boolean, if True, include anisotropic atomic displacement parameters.
    - unified_Uiso: Boolean, if True, unify Uiso values for atoms of the same element.
    - sgoffset: List of three floats, offset applied to the origin of symmetry operations (default: [0, 0, 0]).

    Returns:
    - fit: FitRecipe object, the initialized fitting recipe.
    """
    fit = FitRecipe()
    fit.addContribution(cpdf)

    #------------------ 1 ------------------#
    # Generate a phase equation. Pre-defines scale factors s* parameters for each phase.
    phase_equation = ''
    for i, phase in enumerate(cpdf._generators):
        phase_equation += 's_' + str(phase) + '*' + str(phase) + ' + '
    cpdf.setEquation(phase_equation[:-3])
    print('equation:', cpdf.getEquation())

    #------------------ 2b ------------------#
    # add delta2 as a fitting parameter for each phase independently
    for i, phase in enumerate(cpdf._generators):
        fit.addVar(getattr(cpdf, phase).delta2, name='delta2_' + str(phase), value=2.0,
                   tags=['delta2', str(phase), 'delta2_' + str(phase), 'delta'])
        fit.restrain(getattr(cpdf, phase).delta2, lb=0.0, ub=5, scaled=True, sig=0.005)

    #------------------ 4 ------------------#
    # add scale factors s*
    for i, phase in enumerate(cpdf._generators):
        fit.addVar(getattr(cpdf, 's_' + str(phase)), value=0.1, tags=['scale', str(phase), 's_' + str(phase)])
        fit.restrain(getattr(cpdf, 's_' + str(phase)), lb=0.0, scaled=True, sig=0.0005)

    #------------------ 5 ------------------#
    # determine independent parameters based on the space group
    for i, phase in enumerate(cpdf._generators):
        spaceGroup = str(list(ciffile.values())[i][0])
        sgpar = constrainAsSpaceGroup(getattr(cpdf, phase).phase, spaceGroup, sgoffset=sgoffset)

        #------------------ 6 ------------------#
        # add lattice parameters
        for par in sgpar.latpars:
            fit.addVar(par, value=par.value, name=par.name + '_' + str(phase),
                       fixed=False, tags=['lat', str(phase), 'lat_' + str(phase)])

        #------------------ 7a ------------------#
        # atomic displacement parameters ADPs
        if anisotropic:
            getattr(cpdf, phase).stru.anisotropy = True
            print('Adding anisotropic displacement parameters.')
            for par in sgpar.adppars:
                atom = par.par.obj
                atom_label = atom.label
                atom_symbol = atom.element
                # take Uiso initial from detailed_composition
                u0 = detailed_composition.get(atom_symbol, {}).get('Uiso', 0.01)
                name = f"{par.name}_{atom_label}_{phase}"
                tags = ['adp', f"adp_{atom_label}", f"adp_{atom_symbol}_{phase}", f"adp_{phase}", f"adp_{atom_symbol}", str(phase)]
                fit.addVar(par, value=u0, name=name, tags=tags)
                fit.restrain(par, lb=0.0, ub=0.1, scaled=True, sig=0.0005)
        else:
            mapped_adppars = map_sgpar_params(sgpar, 'adppars')
            added_adps = set()
            for par in sgpar.adppars:
                try:
                    atom_symbol = par.par.obj.element
                    parameter_name = par.name
                    atom_label = mapped_adppars[parameter_name][1]
                    added_adps.add(atom_label)
                except Exception:
                    pass
            if unified_Uiso:
                print('Adding isotropic displacement parameters as unified values.')
                getattr(cpdf, phase).stru.anisotropy = False
                # one Uiso per element
                for el, info in detailed_composition.items():
                    u0 = info['Uiso']
                    var = fit.newVar(f"Uiso_{el}_{phase}", value=u0, tags=['adp', el, str(phase)])
                    for atom in getattr(cpdf, phase).phase.getScatterers():
                        if atom.element == el:
                            fit.constrain(atom.Uiso, var)
                            fit.restrain(atom.Uiso, lb=0.0, ub=0.1, scaled=True, sig=0.0005)
            else:
                print('Adding isotropic displacement parameters as independent values.')
                getattr(cpdf, phase).stru.anisotropy = False
                for atom in getattr(cpdf, phase).phase.getScatterers():
                    el = atom.element
                    if atom.name.upper() in added_adps:
                        u0 = detailed_composition.get(el, {}).get('Uiso', 0.01)
                        fit.addVar(atom.Uiso,
                                   value=u0,
                                   name=f"{atom.name}_{phase}",
                                   fixed=False,
                                   tags=['adp', str(phase), f"adp_{phase}", f"adp_{el}"])
                        fit.restrain(atom.Uiso, lb=0.0, ub=0.1, scaled=True, sig=0.0005)

        #------------------ 8 ------------------#
        # atom positions XYZ
        added_params[str(phase)] = set()  # Initialize for each phase
        mapped_xyzpars = map_sgpar_params(sgpar, 'xyzpars')
        for par in sgpar.xyzpars:
            try:
                atom_symbol = par.par.obj.element
                parameter_name = par.name
                mapped_name = mapped_xyzpars[parameter_name][0]
                atom_label = mapped_xyzpars[parameter_name][1]
                name_long = mapped_name + '_' + str(phase)
                tags = ['xyz', 'xyz_' + atom_symbol, 'xyz_' + atom_symbol + '_' + str(phase),
                        'xyz_' + str(phase), str(phase)]
                fit.addVar(par, name=name_long, tags=tags)
                added_params[str(phase)].add(atom_label)
                print(f"Constrained {name_long} at {par.par.value}: atom position added to variables.")
            except Exception:
                pass

    #------------------ 10 ------------------#
    for i, phase in enumerate(cpdf._generators):
        cpdf.registerFunction(sphericalCF, name='sphere_' + str(phase), argnames=['r', 'psize_' + str(phase)])
    phase_equation = ''
    for i, phase in enumerate(cpdf._generators):
        phase_equation += 's_' + str(phase) + '*' + str(phase) + '*sphere_' + str(phase) + ' + '
    cpdf.setEquation(phase_equation[:-3])
    print('equation:', cpdf.getEquation())

    for i, phase in enumerate(cpdf._generators):
        fit.addVar(getattr(cpdf, 'psize_' + str(phase)), value=100.0, fixed=False, tags=['psize', 'psize_' + str(phase), str(phase)])
        fit.restrain(getattr(cpdf, 'psize_' + str(phase)), lb=0.0, scaled=True, sig=0.1)

    # show diagnostic information
    fit.fithooks[0].verbose = 2

    #------------------ 11 ------------------#
    global global_bond_vectors
    for phase in cpdf._generators:
        print(f"Calculating bond vectors for {phase}")
        bond_vectors = get_polyhedral_bond_vectors(getattr(cpdf, phase).phase)
        global_bond_vectors[phase] = bond_vectors

    return fit



#-----------------------------------------------------------------------------
def modify_fit(fit, cpdf, spacegroup, sgoffset=[0, 0, 0]):
    """
    Modify the fitting recipe by updating the space group and regenerating atomic position variables,
    constraints, and restraints. This function adds new atom positions from a lower symmetry structure,
    but retains original lattice constants

    Parameters:
    - fit: FitRecipe object, the current fitting recipe containing refinement variables and constraints.
    - cpdf: PDFContribution object, whose generators (“phases”) we’ll re‐constrain.
    - spacegroup: List of new space groups to apply to the structure, one space group element per phase
    - sgoffset: List of three floats, offset applied to the origin of the space group symmetry operations
      (default: [0, 0, 0]).

    Returns:
    - fit: Updated FitRecipe object with the new space group constraints and variables.
    """
    global added_params

    # Step 1: Read out all atom position variables from the fit
    old_xyz_vars = {}
    for name in fit.names:
        if name.startswith('x_') or name.startswith('y_') or name.startswith('z_'):
            var_value = getattr(fit, name).value
            old_xyz_vars[name] = var_value

    # Step 3: Delete the old variables and clear added_params for each phase
    for name in old_xyz_vars.keys():
        
        #Deep cleaning...scrub...scrub
        fit.unconstrain(getattr(fit, name))
        # fit.clearConstraints(getattr(fit, name))
        # fit.clearRestraints(getattr(fit, name))
        fit.delVar(getattr(fit, name))
        print(f"{name}: old variable deleted")

    # Clear added_params for each phase
    for phase in cpdf._generators:
        added_params[str(phase)] = set()

    # Step 4: remove rigid body restrictions
    for name in dir(fit):
        if name.startswith('bond_') or name.startswith('angle_') or name.startswith('dihedral_'):
            try:
                fit.unconstrain(getattr(fit, name))
                # fit.clearConstraints(getattr(fit, name))
                # fit.clearRestraints(getattr(fit, name))
                fit.delVar(getattr(fit, name))
                print(f"{name}: old variable deleted")
            except Exception:
                pass

    # Step 5: Apply the new space group and generate new variables
    for i, phase in enumerate(cpdf._generators):
        try:
            sgpar = constrainAsSpaceGroup(getattr(cpdf, phase).phase, spacegroup[i])
            sgpar._clearConstraints()
        except Exception as e:
            print(f"Error in applying space group to phase {phase}: {e}")
            continue

        mapped_xyzpars = map_sgpar_params(sgpar, 'xyzpars')
        for par in sgpar.xyzpars:
            try:
                atom_symbol = par.par.obj.element
                parameter_name = par.name
                mapped_name = mapped_xyzpars[parameter_name][0]
                atom_label = mapped_xyzpars[parameter_name][1]
                name_long = mapped_name + '_' + str(phase)

                tags = ['xyz', 'xyz_' + atom_symbol, 'xyz_' + atom_symbol + '_' + str(phase), 'xyz_' + str(phase), str(phase)]
                old_value = old_xyz_vars.get(name_long, par.par.value)
                fit.addVar(par, value=old_value, name=name_long, tags=tags)
                added_params[str(phase)].add(atom_label)
                print(f"Constrained {name_long} at {old_value}: atom position added to variables.")
            except Exception as e:
                print(f"Error processing {parameter_name} in phase {phase}: {e}")

    # Step 7: Enforce Pseudo-Cubic Constraints for Lattice Parameters and anisotropic ADPs
    old_lattice_vars = {}
    for name in fit.names:
        if name.startswith(('a_', 'b_', 'c_', 'alpha_', 'beta_', 'gamma_')):
            var_value = getattr(fit, name).value
            old_lattice_vars[name] = var_value

    for name in old_lattice_vars.keys():
        try:
            fit.unconstrain(getattr(fit, name))
            # fit.clearConstraints(getattr(fit, name))
            # fit.clearRestraints(getattr(fit, name))
            fit.delVar(getattr(fit, name))
            print(f"{name}: old variable deleted")
        except Exception:
            pass

    for phase in cpdf._generators:
        spaceGroup = str(list(ciffile.values())[0][0])
        sgpar = constrainAsSpaceGroup(getattr(cpdf, phase).phase, spaceGroup, sgoffset=sgoffset)
        for par in sgpar.latpars:
            name = par.name + '_' + str(phase)
            try:
                old_value = old_lattice_vars[name]
                fit.addVar(par, value=old_value, name=name, fixed=False, tags=['lat', str(phase), 'lat_' + str(phase)])
                print(f"Constrained {name} at {old_value}.")
            except Exception:
                pass

    # strip out any constraint whose .par is None
    fit._oconstraints[:] = [c for c in fit._oconstraints if c.par is not None]
    return fit


#-----------------------------------------------------------------------------
def rebuild_cpdf(old_cpdf, r_obs, g_obs, myrange, myrstep, ncpu, pool, name='cpdf2'):
    """
    Build a new PDFContribution using the same Structure objects already present
    in `old_cpdf`, instead of re‐loading from CIFs.

    Parameters
    ----------
    old_cpdf : PDFContribution
        An existing PDFContribution whose PDFGenerators hold refined structures
        (accessible via `old_cpdf.PhaseX.phase.stru`).
    r_obs, g_obs : numpy.ndarray
        The new observed PDF data arrays (r and G(r)).
    myrange : (float, float)
        Tuple (rmin, rmax) specifying the range for PDF calculation.
    myrstep : float
        Step size (Δr) for PDF evaluation.
    ncpu : int
        Number of CPU cores to assign to each new PDFGenerator.
    pool : multiprocessing.Pool
        Pool instance for parallel execution (passed to `.parallel(...)`).
    name : str, optional
        Identifier for the new PDFContribution (default: "cpdf_rebuilt").

    Returns
    -------
    PDFContribution
        A fresh PDFContribution whose PDFGenerators each incorporate the exact
        same `diffpy.Structure` objects from `old_cpdf`.
    """
    cpdf_new = PDFContribution(name)
    cpdf_new.setScatteringType('X')

    try:
        global_qmax = old_cpdf.getQmax()
        cpdf_new.setQmax(global_qmax)
        print(f"[DEBUG] Copied global Qmax = {global_qmax}")
    except Exception:
        print("[DEBUG] old_cpdf has no getQmax() or Qmax transfer failed.")

    cpdf_new.profile.setObservedProfile(r_obs.copy(), g_obs.copy())
    print(f"[DEBUG] Assigned observed profile: {len(r_obs)} points")

    cpdf_new.setCalculationRange(xmin=myrange[0], xmax=myrange[1], dx=myrstep)
    print(f"[DEBUG] Set calculation range: r ∈ [{myrange[0]}, {myrange[1]}], Δr = {myrstep}")

    for phase_name in old_cpdf._generators:
        print(f"[DEBUG] Rebuilding phase generator: '{phase_name}'")
        old_gen = getattr(old_cpdf, phase_name)
        try:
            old_structure = old_gen.phase.stru.copy()
        except Exception:
            print(f"[WARNING] Could not access Structure for '{phase_name}'; skipping.")
            continue

        new_gen = PDFGenerator(str(phase_name))

        try:
            qmin = old_gen._calc.qmin
            new_gen.setQmin(qmin)
            print(f"[DEBUG]  - Copied qmin = {qmin}")
        except Exception:
            print(f"[DEBUG]  - No qmin to copy for '{phase_name}'")

        try:
            qmax = old_gen._calc.qmax
            new_gen.setQmax(qmax)
            print(f"[DEBUG]  - Copied qmax = {qmax}")
        except Exception:
            print(f"[DEBUG]  - No qmax to copy for '{phase_name}'")

        try:
            eval_type = old_gen._calc.evaluatortype
            new_gen._calc.evaluatortype = 'OPTIMIZED'
        except Exception:
            print(f"[DEBUG]  - Could not copy evaluator type for '{phase_name}'")

        try:
            periodic_flag = old_gen.phase.periodic
        except Exception:
            periodic_flag = True

        new_gen.setStructure(old_structure, periodic=periodic_flag)
        print(f"[DEBUG]  - Assigned existing Structure (periodic={periodic_flag})")

        try:
            qdamp_val = old_gen.qdamp.value
            new_gen.qdamp.value = qdamp_val
            print(f"[DEBUG]  - Copied qdamp = {qdamp_val}")
        except Exception:
            print(f"[DEBUG]  - No qdamp to copy for '{phase_name}'")

        try:
            qbroad_val = old_gen.qbroad.value
            new_gen.qbroad.value = qbroad_val
            print(f"[DEBUG]  - Copied qbroad = {qbroad_val}")
        except Exception:
            print(f"[DEBUG]  - No qbroad to copy for '{phase_name}'")

        new_gen.parallel(ncpu=ncpu, mapfunc=pool.map)
        print(f"[DEBUG]  - Enabled parallel with ncpu={ncpu}")

        cpdf_new.addProfileGenerator(new_gen)
        print(f"[DEBUG]  - Added '{phase_name}' generator to new contribution.")

    try:
        res_eq = old_cpdf.getResidualEquation()
        cpdf_new.setResidualEquation(res_eq)
        print(f"[DEBUG] Copied residual equation = '{res_eq}'")
    except Exception:
        print("[DEBUG] No residual equation to copy or transfer failed.")

    n_old = len(old_cpdf._generators)
    n_new = len(cpdf_new._generators)
    print(f"[DEBUG] Completed rebuild: {n_old} -> {n_new} generators")

    return cpdf_new

#-----------------------------------------------------------------------------
def refinement_basic_with_initial(
        fit_old,
        cpdf_new,
        spaceGroups,
        anisotropic,
        unified_Uiso,
        sgoffset=[0, 0, 0],
        recalculate_bond_vectors=False
    ):
    """
    Create a “basic” fitting recipe for `cpdf_new`, but initialize every variable
    with the value it had in `fit_old`, if it exists there.

    Parameters:
    -----------
    fit_old : FitRecipe
        The FitRecipe object from the previous round of refinement. Any variable
        whose name appears in both `fit_old` and the new recipe will have its
        .value copied from `fit_old` into the new FitRecipe.
    cpdf_new : PDFContribution
        The PDFContribution object for the “new” data (e.g. cpdf2). This will be
        added to the new FitRecipe before any variables are created.
    spaceGroups : list of str
        A list of space‐group strings, one per phase in `cpdf_new._generators`.
    anisotropic : bool
        If True, enable anisotropic ADPs for every phase. Otherwise, add isotropic
        Uiso variables (either unified by element or independent, according to
        `unified_Uiso`).
    unified_Uiso : bool
        If True (and `anisotropic=False`), create one shared Uiso per element per phase
        (e.g. Uiso_O_phase, Uiso_V_phase, Uiso_Zr_phase). If False, create an independent
        Uiso variable for each atom in each phase.
    sgoffset : list of 3 floats, optional
        Passed directly to constrainAsSpaceGroup(...). Default is [0, 0, 0].
    recalculate_bond_vectors : bool, optional
        If True, recalculate global bond‐vector dictionaries after building all variables
        (so that rigid‐body constraints can use up‐to‐date bond vectors). If False (default),
        skip this step.

    Returns:
    --------
    fit_new : FitRecipe
        A brand‐new FitRecipe built exactly as in `refinement_basic()`, but with:
        1) the same variables (scale factors, Δ2, lattice, ADPs, xyz, psize) created on
           `cpdf_new` that `refinement_basic()` would create, and
        2) every variable that also existed in `fit_old` having its `.value` initialized
           to the same value it had in `fit_old`.
    """
    print("\n======================================")
    print("[INFO] Building a new basic refinement recipe with initial values from previous fit")
    print("======================================\n")

    # STEP 1: Build the brand‐new “basic” recipe exactly as refinement_basic() does
    fit_new = FitRecipe()
    fit_new.addContribution(cpdf_new)
    print("[INFO] Added PDFContribution to new FitRecipe")

    # 1a) Phase equation (scale factors)
    phase_equation = ""
    for phase in cpdf_new._generators:
        phase_equation += f"s_{phase}*{phase} + "
    cpdf_new.setEquation(phase_equation[:-3])
    print(f"[INFO] Set phase equation: {cpdf_new.getEquation()}")

    # 1b) Add Δ2 for each phase independently
    for phase in cpdf_new._generators:
        fit_new.addVar(
            getattr(cpdf_new, phase).delta2,
            name=f"delta2_{phase}",
            value=2.0,
            tags=["delta2", str(phase), f"delta2_{phase}", "delta"]
        )
        fit_new.restrain(
            getattr(cpdf_new, phase).delta2,
            lb=0.0, ub=5.0, scaled=True, sig=0.005
        )

    # 1c) Add scale factors s_phase
    for phase in cpdf_new._generators:
        fit_new.addVar(
            getattr(cpdf_new, f"s_{phase}"),
            value=0.1,
            tags=["scale", str(phase), f"s_{phase}"]
        )
        fit_new.restrain(
            getattr(cpdf_new, f"s_{phase}"),
            lb=0.0, scaled=True, sig=0.0005
        )

    # 1d) For each phase, apply the provided space group and add lattice, ADP, xyz
    for i, phase in enumerate(cpdf_new._generators):
        spaceGroup = spaceGroups[i]
        print(f"\n[INFO] Applying space group '{spaceGroup}' to phase '{phase}'")
        sgpar = constrainAsSpaceGroup(
            getattr(cpdf_new, phase).phase,
            spaceGroup,
            sgoffset=sgoffset
        )

        # — add lattice parameters (latpars)
        for par in sgpar.latpars:
            fit_new.addVar(
                par,
                value=par.value,
                name=f"{par.name}_{phase}",
                fixed=False,
                tags=["lat", str(phase), f"lat_{phase}"]
            )

        # — add ADPs
        if anisotropic:
            getattr(cpdf_new, phase).stru.anisotropy = True
            print(f"[INFO]   Enabling anisotropic ADPs for phase '{phase}'")
            for par in sgpar.adppars:
                atom = par.par.obj
                atom_label  = atom.label
                atom_symbol = atom.element
                # pull Uiso directly from detailed_composition
                u0 = detailed_composition[atom_symbol]['Uiso']
                name = f"{par.name}_{atom_label}_{phase}"
                tags = [
                    "adp",
                    f"adp_{atom_label}",
                    f"adp_{atom_symbol}_{phase}",
                    f"adp_{phase}",
                    f"adp_{atom_symbol}",
                    str(phase)
                ]
                fit_new.addVar(par, value=u0, name=name, tags=tags)
                fit_new.restrain(
                    par,
                    lb=0.0, ub=0.1, scaled=True, sig=0.0005
                )
        else:
            # isotropic Uiso handling
            mapped_adppars = map_sgpar_params(sgpar, "adppars")
            added_adps = set()
            for par in sgpar.adppars:
                try:
                    atom_label = mapped_adppars[par.name][1]
                    added_adps.add(atom_label)
                except Exception:
                    pass

            if unified_Uiso:
                getattr(cpdf_new, phase).stru.anisotropy = False
                print(f"[INFO]   Using unified isotropic Uiso for phase '{phase}'")
                # one Uiso per element
                for el, info in detailed_composition.items():
                    u0 = info['Uiso']
                    var = fit_new.newVar(
                        f"Uiso_{el}_{phase}",
                        value=u0,
                        tags=["adp", el, str(phase)]
                    )
                    for atom in getattr(cpdf_new, phase).phase.getScatterers():
                        if atom.element == el:
                            fit_new.constrain(atom.Uiso, var)
                            fit_new.restrain(
                                atom.Uiso,
                                lb=0.0, ub=0.1, scaled=True, sig=0.0005
                            )
            else:
                getattr(cpdf_new, phase).stru.anisotropy = False
                print(f"[INFO]   Using independent isotropic Uiso for phase '{phase}'")
                for atom in getattr(cpdf_new, phase).phase.getScatterers():
                    if atom.name.upper() in added_adps:
                        el = atom.element
                        u0 = detailed_composition[el]['Uiso']
                        var_name = f"{atom.name}_{phase}"
                        fit_new.addVar(
                            atom.Uiso,
                            value=u0,
                            name=var_name,
                            fixed=False,
                            tags=["adp", str(phase), f"adp_{phase}", f"adp_{el}"]
                        )
                        fit_new.restrain(
                            atom.Uiso,
                            lb=0.0, ub=0.1, scaled=True, sig=0.0005
                        )

        # — add atomic xyz parameters (xyzpars)
        mapped_xyzpars = map_sgpar_params(sgpar, "xyzpars")
        added_params[str(phase)] = set()
        print(f"[INFO]   Adding atomic position variables (xyzpars) for phase '{phase}'")
        for par in sgpar.xyzpars:
            try:
                atom_symbol = par.par.obj.element
                mapped_name = mapped_xyzpars[par.name][0]
                atom_label  = mapped_xyzpars[par.name][1]
                name_long   = f"{mapped_name}_{phase}"
                tags = [
                    "xyz",
                    f"xyz_{atom_symbol}",
                    f"xyz_{atom_symbol}_{phase}",
                    f"xyz_{phase}",
                    str(phase)
                ]
                fit_new.addVar(par, name=name_long, tags=tags)
                added_params[str(phase)].add(atom_label)
            except Exception:
                pass

    # 1e) Register spherical‐envelope (“psize_”) for each phase
    phase_equation = ""
    for phase in cpdf_new._generators:
        cpdf_new.registerFunction(
            sphericalCF,
            name=f"sphere_{phase}",
            argnames=["r", f"psize_{phase}"]
        )
        phase_equation += f"s_{phase}*{phase}*sphere_{phase} + "
    cpdf_new.setEquation(phase_equation[:-3])
    print(f"\n[INFO] Updated phase equation with spherical envelope: {cpdf_new.getEquation()}")

    for phase in cpdf_new._generators:
        fit_new.addVar(
            getattr(cpdf_new, f"psize_{phase}"),
            value=100.0,
            fixed=False,
            tags=["psize", f"psize_{phase}", str(phase)]
        )
        fit_new.restrain(
            getattr(cpdf_new, f"psize_{phase}"),
            lb=0.0, scaled=True, sig=0.1
        )

    # 1f) Recalculate bond‐vectors for each phase if requested
    if recalculate_bond_vectors:
        global global_bond_vectors
        for phase in cpdf_new._generators:
            bond_vectors = get_polyhedral_bond_vectors(getattr(cpdf_new, phase).phase)
            global_bond_vectors[phase] = bond_vectors
            print(f"[INFO] Recalculated bond vectors for phase '{phase}'")

    # STEP 2: Copy every matching .value from fit_old → fit_new (excluding xyz)
    print("\n[INFO] Copying initial values from previous fit into the new recipe")
    for name in fit_old.names:
        if name.startswith(("x_", "y_", "z_")) or "xyz" in name:
            continue
        if name in fit_new.names:
            try:
                oldval = getattr(fit_old, name).value
                getattr(fit_new, name).value = oldval
                print(f"[INFO]   Copied '{name}' = {oldval}")
            except Exception:
                pass

    print("\n[INFO] Completed building new FitRecipe with initial values\n")

    # Enable verbose residual output (SR‐Fit will print residuals each iteration)
    if fit_new.fithooks:
        fit_new.fithooks[0].verbose = 2

    return fit_new
#-----------------------------------------------------------------------------
def copy_phase_structure(cpdf_target, cpdf_source, phase_index=0):
    """
    Copy x,y,z positions of matching atoms from cpdf_source to cpdf_target
    for a given phase (default Phase0), printing each change.

    Parameters
    ----------
    cpdf_target : PDFContribution
        The “main” PDFContribution you want to update.
    cpdf_source : PDFContribution
        The PDFContribution containing the special (refined) structure.
    phase_index : int, optional
        Which phase to copy (0 → "Phase0", 1 → "Phase1", …).
    """
    phase_name = f"Phase{phase_index}"
    gen_tgt = getattr(cpdf_target, phase_name)
    gen_src = getattr(cpdf_source, phase_name)

    # Prepare source scatterers
    src_scats = gen_src.stru
    # Assign unique labels to avoid collisions (element + index)
    src_scats.assignUniqueLabels()
    # Build a lookup map from scatterer label to object
    src_map = {atom.label: atom for atom in src_scats}

    n_copied = 0
    # Prepare target scatterers similarly
    gen_tgt.stru.assignUniqueLabels()
    for atom in gen_tgt.stru:
        label = atom.label
        if label in src_map:
            # Record old coordinates before updating
            old = (atom.x, atom.y, atom.z)
            # Get new coordinates from source
            src = src_map[label]
            new = (src.x, src.y, src.z)
            # Apply new coordinates to target scatterer
            atom.x = new[0]
            atom.y = new[1]
            atom.z = new[2]
            # Print change for traceability
            print(f"Atom {label}: ({old[0]:.5f}, {old[1]:.5f}, {old[2]:.5f}) → "
                  f"({new[0]:.5f}, {new[1]:.5f}, {new[2]:.5f})")
            n_copied += 1

    # Summary of copied atoms
    print(f"Copied xyz of {n_copied} atoms from {phase_name} of source → target.")


#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def compare_fits(fitA, fitB, tol=1e-8):
    """
    Compare two FitRecipe objects and print any differences in:
      1) Variable sets (names present in one but not the other)
      2) Variable values for names common to both
      3) Restraints (lb, ub, sig) on each parameter

    Only differences are printed. `tol` is the tolerance for comparing numeric values.

    Parameters:
    -----------
    fitA, fitB : FitRecipe
        The two FitRecipe instances to compare.
    tol : float, optional
        Tolerance for comparing numeric values (default: 1e-8).

    Returns:
    --------
    None; prints differences to stdout.
    """
    from math import isclose

    def gather_restraints(fit):
        """
        Build a dict mapping parameter‐name → (lb, ub, sig) for every restraint in fit._oconstraints.
        If a parameter has multiple restraints, only the first is recorded.
        """
        restraints = {}
        for c in fit._oconstraints:
            pname = c.par.name
            # Some constraints may not have lb/ub/sig attributes; handle gracefully
            lb = getattr(c, "lb", None)
            ub = getattr(c, "ub", None)
            sig = getattr(c, "sig", None)
            if pname not in restraints:
                restraints[pname] = (lb, ub, sig)
        return restraints

    # 1) Compare variable name sets
    namesA = set(fitA.names)
    namesB = set(fitB.names)

    onlyA = sorted(namesA - namesB)
    onlyB = sorted(namesB - namesA)
    if onlyA:
        print("Variables only in fitA:")
        for n in onlyA:
            print(f"  {n}")
        print()
    if onlyB:
        print("Variables only in fitB:")
        for n in onlyB:
            print(f"  {n}")
        print()

    # 2) Compare values for names common to both
    common = sorted(namesA & namesB)
    diffs = []
    for name in common:
        valA = getattr(fitA, name).value
        valB = getattr(fitB, name).value
        # If either value is None or not numeric, compare directly
        if valA is None or valB is None:
            if valA != valB:
                diffs.append((name, valA, valB))
        else:
            try:
                if not isclose(valA, valB, rel_tol=tol, abs_tol=tol):
                    diffs.append((name, valA, valB))
            except Exception:
                # In case valA or valB is not numeric
                if valA != valB:
                    diffs.append((name, valA, valB))

    if diffs:
        print("Variable value differences (name: fitA → fitB):")
        for name, a, b in diffs:
            print(f"  {name}: {a} → {b}")
        print()

    # 3) Compare restraints (lb, ub, sig)
    resA = gather_restraints(fitA)
    resB = gather_restraints(fitB)

    keysA = set(resA.keys())
    keysB = set(resB.keys())

    onlyResA = sorted(keysA - keysB)
    onlyResB = sorted(keysB - keysA)
    if onlyResA:
        print("Restraints only in fitA:")
        for pname in onlyResA:
            lb, ub, sig = resA[pname]
            print(f"  {pname}: lb={lb}, ub={ub}, sig={sig}")
        print()
    if onlyResB:
        print("Restraints only in fitB:")
        for pname in onlyResB:
            lb, ub, sig = resB[pname]
            print(f"  {pname}: lb={lb}, ub={ub}, sig={sig}")
        print()

    commonRes = sorted(keysA & keysB)
    res_diffs = []
    for pname in commonRes:
        lbA, ubA, sigA = resA[pname]
        lbB, ubB, sigB = resB[pname]
        # Compare each numeric field with tolerance; treat None separately
        def diff_field(a, b):
            if a is None or b is None:
                return a != b
            try:
                return not isclose(a, b, rel_tol=tol, abs_tol=tol)
            except Exception:
                return a != b

        if diff_field(lbA, lbB) or diff_field(ubA, ubB) or diff_field(sigA, sigB):
            res_diffs.append((pname, (lbA, ubA, sigA), (lbB, ubB, sigB)))

    if res_diffs:
        print("Restraint differences (parameter: fitA(lb,ub,sig) → fitB(lb,ub,sig)):")
        for pname, (lA, uA, sA), (lB, uB, sB) in res_diffs:
            print(f"  {pname}: ({lA}, {uA}, {sA}) → ({lB}, {uB}, {sB})")
        print()

    if not (onlyA or onlyB or diffs or onlyResA or onlyResB or res_diffs):
        print("No differences found between the two FitRecipe objects.")
        
#=============================================================================
# # Simulate PDFs
# # =============================================================================
def simulate_pdf_workflow(
    cif_directory,
    ciffile,
    xrd_directory,
    powder_data_file,
    composition,
    qdamp,
    qbroad,
    qmax,
    r_range,
    r_step,
    optimized_params,
    default_Uiso,
    fitting_range,
    csv_filename,
    output_results_path='',
    font_size=14,
    label_font_size=20
):
    """
    Generate and simulate PDFs from experimental XRD data and CIF structures,
    apply spherical envelope functions, inject optimized parameters,
    evaluate fit, save results, and visualize summaries.
    """
    # Step 1: Generate a “dummy” observed PDF for overlay comparison
    r0_simulated, g0_simulated, cfg_simulated = generatePDF(
        xrd_directory + powder_data_file,
        composition,
        qmin    = 0.0,
        qmax    = qmax,
        myrange = r_range,
        myrstep = r_step
    )

    # Step 2: Create the PDFContribution object for simulation-only data
    cpdf_simulated = phase(
        r0_simulated, g0_simulated, cfg_simulated,
        cif_directory, ciffile,
        r_range, r_step,
        qdamp, qbroad
    )

    # Step 3: Register spherical envelope for each phase and build the equation
    phase_equation = ""
    for ph in cpdf_simulated._generators:
        cpdf_simulated.registerFunction(
            sphericalCF,
            name     = f"sphere_{ph}",
            argnames = ["r", f"psize_{ph}"]
        )
        phase_equation += f"s_{ph}*{ph}*sphere_{ph} + "
    cpdf_simulated.setEquation(phase_equation.rstrip(" + "))

    # Step 4: Inject optimized fit parameters exactly as in initial code
    for ph, params in optimized_params.items():
        # Set scale factor
        getattr(cpdf_simulated, f"s_{ph}").value = params['s']
        # Set particle size parameter
        setattr(cpdf_simulated, f"psize_{ph}", params['psize'])
        # Set delta2 on the phase object
        getattr(cpdf_simulated, ph).delta2.value = params['delta2']

    # Step 5: Set isotropic atomic displacement parameters (Uiso) by element
    for ph in cpdf_simulated._generators:
        struct = getattr(cpdf_simulated, ph).phase
        for atom in struct.getScatterers():
            if atom.element in default_Uiso:
                atom.Uiso.value = default_Uiso[atom.element]

    # Step 6: Evaluate over fitting range, compute Rw, and save CSV
    evaluate_and_plot(
        cpdf_simulated,
        fitting_range = fitting_range,
        csv_filename  = output_results_path+'/'+csv_filename
    )

    # Step 7: Generate bond/angle statistics and save plots
    for phase_name in cpdf_simulated._generators:
        phase_obj = getattr(cpdf_simulated, phase_name).phase
        visualize_fit_summary(
            cpdf_simulated,
            phase_obj,
            output_results_path + f"/{phase_name}_",
            font_size=font_size,
            label_font_size=label_font_size
        )

    return cpdf_simulated
        
#=============================================================================        
# =============================================================================
#                             INITIALIZATION
# =============================================================================
syst_cores = multiprocessing.cpu_count()
cpu_percent = psutil.cpu_percent()
avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
ncpu = int(np.max([1, avail_cores]))
pool = Pool(processes=ncpu)

added_params = {}
global_bond_vectors = {}

checkpoint_file = "checkpoint.pkl"
start_step, fit0, cpdf = load_checkpoint(checkpoint_file)

if fit0 is None or cpdf is None:
    r0, g0, cfg = generatePDF(
    xrd_directory + mypowderdata,
    composition,
    qmin=0.0,
    qmax=qmax,
    myrange=myrange,
    myrstep=myrstep
)

    cpdf = phase(r0, g0, cfg, cif_directory, ciffile, myrange, myrstep, qdamp, qbroad)
    
    fit0 = refinement_basic(
        cpdf,
        anisotropic=anisotropic,
        unified_Uiso=unified_Uiso,
        sgoffset=sgoffset
    )


#----------------------------------------------------------------------
#                       SPECIAL STRUCTURE MODIFICATION
#----------------------------------------------------------------------
# Here we load a “special” refined structure from a previous PDF refinement
# (e.g. the result of refining Phase0 at 25 °C in space group P1, replicate (1×1×1)).
# This refined CIF contains updated atomic positions that broke higher symmetry
# constraints in order to capture subtle distortions at this temperature.
#cif_directory_special = 'fits/ZirconiumVanadate25Cperiodic/18032025_071408/'  # folder with the refined‐structure CIFs
# # Define the special CIF: filename → [space group, periodicity, supercell dims]
# # ‘P1’ means no symmetry constraints, so all atom positions were independently refined.
# # True indicates the structure is treated as periodic; (1,1,1) means no supercell expansion.
# ciffile_special = {'Phase0_6.cif': ['P1', True, (1, 1, 1)]}

# # Build a PDFContribution for this “special” structure using the same PDF data (r0, g0).
# # This gives us access to the fully‐refined atomic coordinates under P1.
# cpdf_special = phase(
#     r0, g0, cfg,
#     cif_directory_special, ciffile_special,
#     myrange, myrstep,
#     qdamp, qbroad
# )

# # At this point, we have two contributions:
# #  - cpdf: the original model with higher‐symmetry CIF(s)
# #  - cpdf_special : the P1‐refined model with individually‐refined atom positions
# # We now transfer the refined Phase0 atomic coordinates from cpdf_special → cpdf,
# # so that subsequent refinements begin from these updated positions.
# copy_phase_structure(cpdf, cpdf_special, phase_index=0)

# =============================================================================
#                               FITTING STEPS
# =============================================================================

# Initialize RefinementStep namedtuple to store refinement parameters 
RefinementStep = namedtuple('RefinementStep', [
    'step_index', 'spacegroup', 'constrain_bonds', 'constrain_angles', 'constrain_dihedrals'
])

refinement_steps = [
    RefinementStep(0, ['Pa-3'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep(1, ['Pa-3'], (True, 0.0001), (True, 0.0001), (False, 0.001)),
    RefinementStep(2, ['P213'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep(3, ['P23'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep(4, ['P23'], (True, 0.0001), (True, 0.0001), (False, 0.001)),
    RefinementStep(5, ['P1'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep(6, ['P1'], (True, 0.0001), (True, 0.0001), (False, 0.001)),
]

# Define common parameters
fitting_order = ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
fitting_range = [1.5, 27]
residualEquation = 'resv'

# ======================== Start fits with checkpointing =====================

for step in refinement_steps:
    if step.step_index < start_step:
        continue  # Already completed this step

    fit0 = modify_fit(fit0, cpdf, step.spacegroup, sgoffset=sgoffset)
    fit0 = refinement_RigidBody(
        fit0, cpdf, 
        step.constrain_bonds, 
        step.constrain_angles, 
        step.constrain_dihedrals, 
        adaptive=False
    )
    fit_me(
        step.step_index, fitting_range, myrstep, fitting_order, 
        fit0, cpdf, residualEquation, output_results_path, **convergence_options
    )
    # --- Save checkpoint after each step ---
    save_checkpoint(step.step_index + 1, fit0, cpdf, checkpoint_file)
    print(f"CHECKPOINT Saved at step {step.step_index + 1}")

# =============================================================================
#                             FINALIZE RESULTS
# =============================================================================
finalize_results(cpdf, fit0, output_results_path, myrange, myrstep)



# =============================================================================
#                   CONTINUE FITTING WITH DIFFERENT DATA
# =============================================================================

# =============================== XRD Data ====================================
mypowderdata = 'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat'

# =============================== Output Setup ================================
time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
output_results_path = fit_directory + project_name + str(time_stamp) + '/'
if not os.path.isdir(output_results_path):
    os.makedirs(output_results_path)

checkpoint_file_2 = "checkpoint_2.pkl"
start_step_2, fit1, cpdf2 = load_checkpoint(checkpoint_file_2)

if fit1 is None or cpdf2 is None:
    # Load the new data    
    r0, g0, cfg = generatePDF(
        xrd_directory + mypowderdata,
        composition,
        qmin=0.0,
        qmax=qmax,
        myrange=myrange,
        myrstep=myrstep
    )
    
    # Build a fresh cpdf2 that reuses the exact same Structures from `cpdf`
    cpdf2 = rebuild_cpdf(
        old_cpdf=cpdf,
        r_obs=r0,
        g_obs=g0,
        myrange=myrange,
        myrstep=myrstep,
        ncpu=ncpu,
        pool=pool,
        name='cpdf2'
    )
    
    # Re‐initialize a brand‐new FitRecipe from cpdf2:
    fit1 = refinement_basic_with_initial(fit0,
        cpdf2, ['Pa-3'],
        anisotropic=anisotropic,
        unified_Uiso=unified_Uiso,
        sgoffset=sgoffset, recalculate_bond_vectors=False)

# Initialize RefinementStep2 namedtuple to store refinement parameters 
RefinementStep2 = namedtuple('RefinementStep', [
    'step_index', 'spacegroup', 'constrain_bonds', 'constrain_angles', 'constrain_dihedrals'
])

refinement_steps2 = [
    RefinementStep2(0, ['Pa-3'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep2(1, ['Pa-3'], (True, 0.0001), (True, 0.0001), (False, 0.001)),
    RefinementStep2(2, ['P213'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep2(3, ['P23'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep2(4, ['P23'], (True, 0.0001), (True, 0.0001), (False, 0.001)),
    RefinementStep2(5, ['P1'], (True, 0.001), (True, 0.001), (False, 0.001)),
    RefinementStep2(6, ['P1'], (True, 0.0001), (True, 0.0001), (False, 0.001)),
]

# Define common parameters
fitting_order = ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
fitting_range = [1.5, 27]
residualEquation = 'resv'

# ======================== Start fits with checkpointing =====================

for step2 in refinement_steps2:
    if step2.step_index < start_step_2:
        continue  # Already completed this step

    fit1 = modify_fit(fit1, cpdf2, step2.spacegroup, sgoffset=sgoffset)
    fit0 = refinement_RigidBody(
        fit1, cpdf2, 
        step2.constrain_bonds, 
        step2.constrain_angles, 
        step2.constrain_dihedrals, 
        adaptive=False
    )
    fit_me(
        step2.step_index, fitting_range, myrstep, fitting_order, 
        fit1, cpdf2, residualEquation, output_results_path, **convergence_options
    )
    # --- Save checkpoint after each step ---
    save_checkpoint(step2.step_index + 1, fit1, cpdf2, checkpoint_file_2)
    print(f"CHECKPOINT Saved at step {step2.step_index + 1}")

# =============================================================================
#                             FINALIZE RESULTS (new data)
# =============================================================================
finalize_results(cpdf2, fit1, output_results_path, myrange, myrstep)

# 


# # # # =============================================================================
# # # # Run simulation workflow with specific parameters (using original code variables)
# # # # =============================================================================

# # # CIF files for simulation
# # cif_directory = 'optimised_PDF_fits_vs_Temp/25C_Phase0_6/'
# # ciffile       = {'opt_25C_Phase0_6.cif': ['P1', True, (1, 1, 1)]}

# # # Experimental PDF data
# # xrd_directory    = 'data/'
# # mypowderdata     = 'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat'
# # composition      = 'O7 V2 Zr1'

# # # Instrumental/sample parameters
# # qdamp   = 2.70577268e-02
# # qbroad  = 2.40376789e-06
# # qmax    = 22.0
# # myrange = (0.0, 80.0)
# # myrstep = 0.05

# # # Optimized parameters from prior refinement
# # optimized_params = {
# #     'Phase0': {
# #         's':      4.92399836e-01,
# #         'psize':  2.66658626e+02,
# #         'delta2': 2.53696631e+00
# #     }
# # }

# # # Default Uiso by element
# # default_Uiso = {'Zr':5.79086780e-04, 'V':3.21503909e-03, 'O':7.21519003e-03}

# # # Fitting and output settings
# # fitting_range       = [1.5, 27]
# # csv_filename        = 'sim_vs_obs.csv'
# # output_results_path = 'resultsSimulations/25C_Phase0_6'
# # if not os.path.isdir(output_results_path):
# #     os.makedirs(output_results_path)

# # # Execute workflow
# # cpdf_simulated = simulate_pdf_workflow(
# #     cif_directory,
# #     ciffile,
# #     xrd_directory,
# #     mypowderdata,
# #     composition,
# #     qdamp,
# #     qbroad,
# #     qmax,
# #     myrange,
# #     myrstep,
# #     optimized_params,
# #     default_Uiso,
# #     fitting_range,
# #     csv_filename,
# #     output_results_path,
# #     font_size=14,
# #     label_font_size=20
# # )        

    # # # =============================================================================
    # # # End of script
    # # # =============================================================================



