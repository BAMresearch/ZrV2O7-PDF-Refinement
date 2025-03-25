"""
Created on Thu Nov 23 12:39:22 2023
Updated on Mar 25 2025

@author: Tomasz Stawski
tomasz.stawski@gmail.com
tomasz.stawski@bam.de

version 1.0.17

# =============================================================================
# DESCRIPTION:
# This script implements a refinement procedure for ZrV2O7-based structures.
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

# =============================================================================
# Import data management packages
# These packages are essential for data manipulation, numerical calculations, 
# and object serialization.
import copy         # For deep and shallow copying of objects.
import numpy as np  # Numerical operations and arrays.
import pandas as pd # Data manipulation and analysis.
import os           # Operating system interface.
import io # Working with streams
import contextlib  # Context manager
import pickle       # For serializing and deserializing Python objects.
from datetime import datetime #for time related operations
from itertools import combinations # Combinatorial functions



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




###############################################################################
#                               ALL THE FUNCTIONS 
###############################################################################


# =============================================================================
# Visualisation and summary functions
# =============================================================================
def plotmyfit(recipe, baseline=-4, ax=None):
    """
    Plot the observed, calculated, and difference (Gobs, Gcalc, Gdiff) PDFs.

    Parameters:
    - recipe: The fitting recipe object containing observed and calculated profiles.
    - baseline: Float, baseline offset for difference plot (default: -4).
    - ax: Matplotlib axis object, optional. If None, a new axis is created.

    Returns:
    - rv: List of matplotlib line objects representing the plots.
    - df: Pandas DataFrame with columns ['x', 'yobs', 'ycalc', 'ydiff'] containing
      the profile data.
    """
    from matplotlib import pyplot
    
    if ax is None:
        ax = pyplot.gca()
    x = recipe.cpdf.profile.x
    yobs = recipe.cpdf.profile.y
    ycalc = recipe.cpdf.evaluate()
    ydiff = yobs - ycalc
    rv = []
    #rv += ax.plot(r0, g0,'o', color = 'black', label='full range' )
    rv += ax.plot(x, yobs, 'o', label='Gobs',
                    markeredgecolor='blue', markerfacecolor='none')
    rv += ax.plot(x, ycalc, color='red', label='Gcalc')
    rv += ax.plot(x, ydiff + baseline, label='Gdiff', color='green')
    rv += ax.plot(x, baseline + 0 * x, linestyle=':', color='black')
    ax.set_xlabel(u'r (Å)')
    ax.set_ylabel(u'G (Å$^{-2}$)')
    #create a dataframe with all the data and fits which can be later saved
    df = pd.DataFrame([x, yobs, ycalc, ydiff])
    df = df.transpose()
    df.columns = ('x', 'yobs', 'ycalc', 'ydiff')
    return rv, df
#----------------------------------------------------------------------------

def visualize_fit_summary(fit, phase, output_plot_path, font_size=14, label_font_size=20):
    """
    Visualize and summarize the fit results, including PDF data, bond lengths, and bond angles.

    Parameters:
    - fit: FitRecipe object, the refined fitting recipe containing the fit results.
    - phase: Structure object, representing the phase used in the refinement.
    - output_plot_path: String, directory path where the output plots and summary will be saved.
    - font_size: Integer, font size for plot text (default: 14).
    - label_font_size: Integer, font size for axis labels and titles (default: 20).

    Returns:
    - None. Saves plots summarizing the fit results to the specified directory.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Helper function to find angle triplets (already updated with V-O-V detection)
    def find_angle_triplets_full(bond_vectors):
        """
        Identify and calculate angle triplets involving bonds, including categories like O-Zr-O, O-V-O,
        Zr-O-V, and V-O-V, based on bond vectors.
    
        Parameters:
        - bond_vectors: Dictionary containing bond information for 'Zr-O', 'V-O', and possibly 'O-O'.
          Each entry includes bond vectors, bond lengths, and associated atom details.
    
        Returns:
        - angle_triplets: List of dictionaries
        """
        angle_triplets = []
        unique_triplets = set()
        bonds_by_central_atom = {}

        # Group Zr-O and V-O bonds by their central atom
        for bond_type in ['Zr-O', 'V-O']:
            for bond in bond_vectors[bond_type]:
                central_label = bond['central_atom']['label']
                if central_label not in bonds_by_central_atom:
                    bonds_by_central_atom[central_label] = []
                bonds_by_central_atom[central_label].append(bond)

        # Detect O-Zr-O and O-V-O angles
        for central_label, bonds in bonds_by_central_atom.items():
            if len(bonds) > 1:
                for i in range(len(bonds)):
                    for j in range(i + 1, len(bonds)):
                        atom1_label = bonds[i]['atom2']['label'] if bonds[i]['atom1']['label'] == central_label else bonds[i]['atom1']['label']
                        atom2_label = bonds[j]['atom2']['label'] if bonds[j]['atom1']['label'] == central_label else bonds[j]['atom1']['label']

                        if atom1_label == atom2_label or {atom1_label, atom2_label, central_label} in unique_triplets:
                            continue

                        vector1 = bonds[i]['vector']
                        vector2 = bonds[j]['vector']
                        angle = calculate_angle(vector1, vector2)
                        angle_category = 'O-Zr-O' if central_label.startswith('ZR') else 'O-V-O'

                        angle_triplets.append({
                            'central_label': central_label,
                            'atom1_label': atom1_label,
                            'atom2_label': atom2_label,
                            'angle': angle,
                            'angle name': (atom1_label, central_label, atom2_label),
                            'angle category': angle_category
                        })
                        unique_triplets.add(frozenset([central_label, atom1_label, atom2_label]))

        # Detect Zr-O-V and V-O-V angles
        oxygen_atoms = {}
        for bond_type in ['Zr-O', 'V-O']:
            for bond in bond_vectors[bond_type]:
                oxygen_label = bond['atom2']['label'] if bond['atom1']['label'] == bond['central_atom']['label'] else bond['atom1']['label']
                if oxygen_label not in oxygen_atoms:
                    oxygen_atoms[oxygen_label] = []
                oxygen_atoms[oxygen_label].append(bond)

        for oxygen_label, bonds in oxygen_atoms.items():
            if len(bonds) > 1:
                for i in range(len(bonds)):
                    for j in range(i + 1, len(bonds)):
                        central_label1 = bonds[i]['central_atom']['label']
                        central_label2 = bonds[j]['central_atom']['label']

                        if central_label1.startswith('ZR') and central_label2.startswith('V'):
                            angle_category = 'Zr-O-V'
                        elif central_label1.startswith('V') and central_label2.startswith('V'):
                            angle_category = 'V-O-V'
                        else:
                            continue

                        vector1 = bonds[i]['vector']
                        vector2 = bonds[j]['vector']
                        angle = calculate_angle(vector1, vector2)

                        triplet_sorted = tuple(sorted([central_label1, oxygen_label, central_label2]))
                        if triplet_sorted not in unique_triplets:
                            angle_triplets.append({
                                'central_label': oxygen_label,
                                'atom1_label': central_label1,
                                'atom2_label': central_label2,
                                'angle': angle,
                                'angle name': (central_label1, oxygen_label, central_label2),
                                'angle category': angle_category
                            })
                            unique_triplets.add(frozenset([central_label1, oxygen_label, central_label2]))

        return angle_triplets

    # Calculate bond vectors and angle triplets
    bond_vectors = get_polyhedral_bond_vectors(phase, zr_o_cutoff=(1.8, 2.8), v_o_cutoff=(1.5, 2.4), max_workers=4)
    angle_triplets = find_angle_triplets_full(bond_vectors)

    # Set global font properties
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = 'Arial'

    # Create the figure with 1 row and 3 panels
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    # 1. Plot the PDF data and the fit
    r = fit.cpdf.profile.x
    g_obs = fit.cpdf.profile.y
    g_calc = fit.cpdf.evaluate()
    g_diff = g_obs - g_calc

    axs[0].plot(r, g_obs, 'o', label='G_obs', markeredgecolor='blue', markerfacecolor='none')
    axs[0].plot(r, g_calc, color='red', label='G_calc')
    axs[0].plot(r, g_diff - 4, color='green', label='G_diff')
    axs[0].axhline(y=-4, color='black', linestyle=':')
    axs[0].set_xlabel('r (Å)')
    axs[0].set_ylabel('G (Å^-2)')
    axs[0].legend()
    axs[0].set_title('PDF Data and Fit')

    # 2. Histogram of Bond Lengths
    zr_o_lengths = [bond['length'] for bond in bond_vectors['Zr-O']]
    v_o_lengths = [bond['length'] for bond in bond_vectors['V-O']]

    sns.histplot(zr_o_lengths, bins=50, kde=True, ax=axs[1], color='blue', label='Zr-O Bonds')
    sns.histplot(v_o_lengths, bins=50, kde=True, ax=axs[1], color='red', label='V-O Bonds')

    axs[1].set_xlabel('Bond Length (Å)')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Zr-O and V-O Bond Length Distribution')
    axs[1].legend()

    # 3. Histogram of Bond Angles
    o_zr_o_angles = [angle['angle'] for angle in angle_triplets if angle['angle category'] == 'O-Zr-O']
    o_v_o_angles = [angle['angle'] for angle in angle_triplets if angle['angle category'] == 'O-V-O']
    zr_o_v_angles = [angle['angle'] for angle in angle_triplets if angle['angle category'] == 'Zr-O-V']
    v_o_v_angles = [angle['angle'] for angle in angle_triplets if angle['angle category'] == 'V-O-V']

    sns.histplot(o_zr_o_angles, bins=np.linspace(70, 180, 50), kde=True, ax=axs[2], color='blue', label='O-Zr-O Angles')
    sns.histplot(o_v_o_angles, bins=np.linspace(70, 180, 50), kde=True, ax=axs[2], color='red', label='O-V-O Angles')
    sns.histplot(zr_o_v_angles, bins=np.linspace(70, 180, 50), kde=True, ax=axs[2], color='green', label='Zr-O-V Angles')
    sns.histplot(v_o_v_angles, bins=np.linspace(70, 180, 50), kde=True, ax=axs[2], color='black', label='V-O-V Angles')

    axs[2].set_xlabel('Angle (°)')
    axs[2].set_ylabel('Count')
    axs[2].set_xlim(50, 180)
    axs[2].set_title('Bond Angle Distributions')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(output_plot_path + 'summary_plot.pdf', dpi=600)
    plt.show()
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
    #export the structures from the cpdf object as cifs
    for phase in cpdf._generators:
        getattr(cpdf,str(phase)).stru.write(output_path + str(phase)+'_'+ str(i) + '.cif', format='cif')
    return
#----------------------------------------------------------------------------
def saveResults(i,fit, output_results_path):
    """
    Save the results of a refinement step, including fitting summary, visualizations,
    and data files.

    Parameters:
    - i: Integer, the step or iteration index used in naming the output files.
    - fit: FitRecipe object, containing the refined model and fit results.
    - output_results_path: String, the directory path where results will be saved.

    Returns:
    - res: FitResults object, which contains a summary of the refinement results.
    """
    res = FitResults(fit)
#a report file from DiffPy
    res.saveResults(output_results_path +'fitting_summary_' + str(i) + '.txt')
#show and save summary figures
    fig0 ,ax0 = subplots()
    fig, df = plotmyfit(fit, ax=ax0)
    fig0.savefig(output_results_path + 'fitting_'+ str(i) + '.png', dpi = 600)
#save all the curves: G(r), fits etc.
    df.to_csv(output_results_path + 'fitting_curve_'+ str(i) + '.csv')
    export_cifs(i,cpdf, output_results_path)
#pickle all the relevant objects
#does not work with parallel processing! For this to work disable this line:
#pdf.parallel(ncpu=ncpu, mapfunc=pool.map)
    
    # #the fit object    
    #     fit_object_file = open(output_results_path+'fit'+str(i)+'.obj', 'wb')
    #     pickle.dump(fit0,fit_object_file)
    #     fit_object_file.close()
        
    # #the PDF object    
    #     cpdf_object_file = open(output_results_path+'cpdf'+str(i)+'.obj', 'wb')
    #     pickle.dump(cpdf,cpdf_object_file)
    #     cpdf_object_file.close()
    
    #Capture the current fit.show() output in a text file (using redirect_stdout)
    fit_show_filename = os.path.join(output_results_path, f"fit_state_{i}.txt")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fit.show()        # No 'out=' parameter
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
        for phase in scale_factors.keys():
            getattr(cpdf, 's_' + phase).value = scale_factors[phase]
            fig0, ax0 = subplots()
            fig, df = plotmyfit(fit, ax=ax0)
            fig0.savefig(output_results_path + f'{phase}_partial_fit.png', dpi=600)
            df.to_csv(output_results_path + f'{phase}_partial.csv')
            getattr(cpdf, 's_' + phase).value = 0

        # Restore the original scale factors
        for phase in scale_factors.keys():
            getattr(cpdf, 's_' + phase).value = scale_factors[phase]

    # Extrapolation of the fit to a full range regardless of the fitting steps
    cpdf.setCalculationRange(myrange[0], myrange[1], myrstep)
    fig0, ax0 = subplots()
    fig, df = plotmyfit(fit, ax=ax0)
    fig0.savefig(output_results_path + '_final_extrapolated_fit.png', dpi=600)
    df.to_csv(output_results_path + '_final_extrapolated_fit.csv')

    print(f"Results finalized and saved to {output_results_path}")
#-----------------------------------------------------------------------------

# =============================================================================
# PDF generation and handling
# =============================================================================
def simulatePDF(file,**config):
    """
    Simulate a PDF from a crystal structure file using the given configuration.

    Parameters:
    - file: String, path to the crystal structure file (e.g., CIF).
    - **config: Additional keyword arguments for PDFCalculator (e.g., rmin, rmax, qmin, qmax).

    Returns:
    - df: Pandas DataFrame with columns ['r', 'g'] representing the simulated PDF.
    """
    
    phase = loadCrystal(file)
    pdf = PDFCalculator(**config)
    r, g = pdf(phase)
    df = pd.DataFrame([r,g])
    df = df.transpose()
    df.columns = ('r', 'g')
    return df
#----------------------------------------------------------------------------
def generatePDF(mypowderdata, composition, qmin = 0.0, qmax = 22, myrange = (1, 100), myrstep = 0.01):
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
                dataformat='QA', rpoly = 1.3,
                rstep=myrstep, rmin=myrange[0], rmax=myrange[1])
    cfg.qmax = cfg.qmaxinst = qmax
    cfg.qmin = qmin
    pg0 = PDFGetter(config=cfg)
    r0, g0 = pg0(filename=mypowderdata)
    
    return r0, g0, cfg

#----------------------------------------------------------------------------
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
    pdfgenerator = PDFGenerator(str(name))
    pdfgenerator.setQmax(qmax)
    pdfgenerator.setQmin(qmin)
    pdfgenerator._calc.evaluatortype = 'OPTIMIZED'
    structure = loadStructure(ciffile)
    
    #allow an expansion to a supercell, which is defined by a 3-element tuple
    structure = expansion.supercell_mod.supercell(structure, (super_cell[0],super_cell[1],super_cell[2]))
    #structure.anisotropy = False
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
    # Initialize the Debye-based PDF generator
    pdfgenerator = DebyePDFGenerator(str(name))

    pdfgenerator.setQmax(qmax)
    pdfgenerator.setQmin(qmin)
    pdfgenerator._calc.evaluatortype = 'OPTIMIZED'
    structure = loadStructure(ciffile)
    
    #allow an expansion to a supercell, which is defined by a 3-element tuple
    structure = expansion.supercell_mod.supercell(structure, (super_cell[0],super_cell[1],super_cell[2]))
    #structure.anisotropy = False
    pdfgenerator.setStructure(structure, periodic=periodic)
    return pdfgenerator


#----------------------------------------------------------------------------
def phase(r0, g0, cfg, cif_directory, ciffile, fitRange, dx, qdamp, qbroad):
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
    cpdf = PDFContribution('cpdf')
    cpdf.setScatteringType('X')
    cpdf.setQmax(cfg.qmax)
    cpdf.profile.setObservedProfile(r0.copy(), g0.copy())
    cpdf.setCalculationRange(xmin=fitRange[0], xmax=fitRange[1], dx=dx)
    # specify an independent  structure model
    for i, file in enumerate(ciffile.keys()):
        print('Phase: ', i, file)
        periodic = str(list(ciffile.values())[i][1])
        super_cell = list(ciffile.values())[i][2]
        print('Phase periodic? ', periodic)
        pdf = contribution('Phase'+str(i), cfg.qmin, cfg.qmax, cif_directory+file, periodic, super_cell)    
        pdf.qdamp.value = qdamp
        pdf.qbroad.value = qbroad
#enable multiprocessing
        pdf.parallel(ncpu=ncpu, mapfunc=pool.map)
        cpdf.addProfileGenerator(pdf)
    cpdf.setResidualEquation('resv')
    #cpdf.setResidualEquation('chiv')
    return cpdf
#-----------------------------------------------------------------------------

# =============================================================================
# Rigig body and aconnectivity
# =============================================================================
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
    angle = np.degrees(np.arccos(np.round(cos_theta, 12)))
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
    import numpy as np

    # Validate non-zero vectors
    def is_zero_vector(vec):
        return np.linalg.norm(vec) < 1e-12

    if is_zero_vector(v1) or is_zero_vector(v2) or is_zero_vector(v3):
        print(f"Zero vector detected: v1={v1}, v2={v2}, v3={v3}")
        return None

    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # Calculate the normals to the planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Validate non-zero normals
    if is_zero_vector(n1) or is_zero_vector(n2):
        print(f"Zero normal vector detected: n1={n1}, n2={n2}")
        return None

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # Calculate dihedral angle
    x = np.dot(n1, n2)
    y = np.dot(v2, np.cross(n1, n2))

    # Handle numerical precision issues with arccos and arctan2
    x = np.clip(x, -1.0, 1.0)  # Ensure the value is within the valid range for arccos
    dihedral_angle = np.degrees(np.arctan2(y, x))

    return dihedral_angle


#----------------------------------------------------------------------------
def get_polyhedral_bond_vectors(phase, zr_o_cutoff=(1.9, 2.2), v_o_cutoff=(1.5, 2.3), max_workers=4):
    """
    Compute bond vectors for Zr-O, V-O, and O-O bonds in a given phase structure,
    considering specified cutoff ranges for Zr-O and V-O bonds.

    Parameters:
    - phase: Object representing the crystal structure of the phase being analyzed.
    - zr_o_cutoff: Tuple of two floats, minimum and maximum distance for Zr-O bonds (default: (2.0, 2.2)).
    - v_o_cutoff: Tuple of two floats, minimum and maximum distance for V-O bonds (default: (1.5, 1.8)).
    - max_workers: Integer, the maximum number of threads for parallel processing (default: 4).

    Returns:
    - bond_vectors: Dictionary containing bond information, structured as:
        {
          'Zr-O': List of dictionaries, each with details of Zr-O bonds,
          'V-O': List of dictionaries, each with details of V-O bonds,
          'O-O': List of dictionaries, each with details of O-O bonds.
        }
    """
    bond_vectors = {'Zr-O': [], 'V-O': [], 'O-O': []}
    
    # Extract lattice constants
    lattice = phase.lattice
    a, b, c = lattice.a.value, lattice.b.value, lattice.c.value

    # Extract positions, elements, and labels from the structure
    scatterers = phase.getScatterers()
    positions = {i: (atom.x.value, atom.y.value, atom.z.value) for i, atom in enumerate(scatterers)}
    elements = {i: atom.element for i, atom in enumerate(scatterers)}
    labels = {i: atom.name.upper() for i, atom in enumerate(scatterers)}
    coordination_groups = {}  # Store O atoms around each central atom

    # Find Zr-O and V-O bonds, storing coordination groups
    with tqdm(total=len(scatterers), desc="Calculating Bond Vectors") as pbar:
        for i, atom in enumerate(scatterers):
            pbar.update(1)
            if elements[i] in ['Zr', 'V']:
                central_atom = elements[i]
                coordination_groups[i] = []  # Store O atom indices coordinated with each Zr or V
                
                for j, neighbor in enumerate(scatterers):
                    if elements[j] == 'O' and i != j:
                        central_pos = np.array([atom.x.value, atom.y.value, atom.z.value])
                        neighbor_pos = np.array([neighbor.x.value, neighbor.y.value, neighbor.z.value])
                        vector = neighbor_pos - central_pos
                        bond_length = np.linalg.norm(vector * np.array([a, b, c]))
                        
                        # Check cutoff for Zr-O or V-O
                        if (central_atom == 'Zr' and zr_o_cutoff[0] <= bond_length <= zr_o_cutoff[1]) or \
                           (central_atom == 'V' and v_o_cutoff[0] <= bond_length <= v_o_cutoff[1]):
                            bond_info = {
                                'central_atom': {
                                    'symbol': central_atom,
                                    'index': i,
                                    'label': labels[i],
                                    'position': positions[i]  # Add central atom position
                                },
                                'atom1': {
                                    'symbol': central_atom,
                                    'index': i,
                                    'label': labels[i],
                                    'position': positions[i]  # Add atom1 position
                                },
                                'atom2': {
                                    'symbol': 'O',
                                    'index': j,
                                    'label': labels[j],
                                    'position': positions[j]  # Add atom2 position
                                },
                                'vector': vector,
                                'length': bond_length,
                                'relative_length': np.linalg.norm(vector)  # Using fractional coordinates
                            }
                            bond_type = 'Zr-O' if central_atom == 'Zr' else 'V-O'
                            bond_vectors[bond_type].append(bond_info)

                            # Append the index of the oxygen atom to the coordination group
                            coordination_groups[i].append(j)

    # Calculate O-O bonds within each coordination group
    for central_index, o_indices in coordination_groups.items():
        for i in range(len(o_indices)):
            for j in range(i + 1, len(o_indices)):
                pos_i = np.array(positions[o_indices[i]])
                pos_j = np.array(positions[o_indices[j]])
                vector = pos_j - pos_i
                length = np.linalg.norm(vector * np.array([a, b, c]))

                bond_info = {
                    'central_atom': {
                        'symbol': elements[central_index],
                        'index': central_index,
                        'label': labels[central_index],
                        'position': positions[central_index]  # Add central atom position
                    },
                    'atom1': {
                        'symbol': 'O',
                        'index': o_indices[i],
                        'label': labels[o_indices[i]],
                        'position': positions[o_indices[i]]  # Add atom1 position
                    },
                    'atom2': {
                        'symbol': 'O',
                        'index': o_indices[j],
                        'label': labels[o_indices[j]],
                        'position': positions[o_indices[j]]  # Add atom2 position
                    },
                    'vector': vector,
                    'length': length,
                    'relative_length': np.linalg.norm(vector)  # Using fractional coordinates
                }
                bond_vectors['O-O'].append(bond_info)

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
    bond_pairs = []  # Initialize an empty list to store valid bond pairs
    phase_added_params = added_params[str(phase)]  # Retrieve phase-specific added parameters for filtering

    # Iterate over each bond type ('Zr-O', 'V-O', 'O-O') and the corresponding bonds
    for bond_type, bonds in bond_vectors.items():
        for bond in bonds:
            # Extract labels for atom1 and atom2
            atom1_label = bond['atom1']['label']
            atom2_label = bond['atom2']['label']
            
            # Check if both atom labels are included in the phase-specific added parameters
            if atom1_label in phase_added_params and atom2_label in phase_added_params:
                # Create a dictionary for the bond pair with relevant bond details
                bond_pair = {
                    'bond_type': bond_type,  # Type of bond ('Zr-O', 'V-O', or 'O-O')
                    'atom1_label': atom1_label,  # Label of the first atom
                    'atom2_label': atom2_label,  # Label of the second atom
                    'atom1_position': bond['atom1']['position'],  # Position of the first atom
                    'atom2_position': bond['atom2']['position'],  # Position of the second atom
                    'central_label': bond['central_atom']['label'],  # Label of the central atom
                    'central_position': bond['central_atom']['position'],  # Position of the central atom
                    'length': bond['length'],  # Bond length in Cartesian coordinates
                    'relative_length': bond['relative_length'],  # Length in fractional coordinates
                    'vector': bond['vector']  # Bond vector from atom1 to atom2
                }
                
                # Add the valid bond pair to the bond_pairs list
                bond_pairs.append(bond_pair)
                
    # Return the list of bond pairs that meet the phase-specific criteria
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
    angle_triplets = []  # List to store unique angle triplets
    unique_triplets = set()  # Set to track unique triplets by their sorted labels

    # Group bonds by central atom to easily access neighbors for each central atom
    bonds_by_central_atom = {}
    for bond in bond_pairs:
        central_label = bond['central_label']
        if central_label not in bonds_by_central_atom:
            bonds_by_central_atom[central_label] = []
        bonds_by_central_atom[central_label].append(bond)

    # Iterate over each central atom to form angle triplets
    for central_label, bonds in bonds_by_central_atom.items():
        # Collect all neighbor atoms around the central atom
        neighbor_atoms = []
        for bond in bonds:
            neighbor_atoms.append((bond['atom1_label'], bond['vector']) if bond['atom1_label'] != central_label else
                                  (bond['atom2_label'], bond['vector']))

        # Form triplets involving the central atom (e.g., O-central-O)
        for atom_combo in combinations(neighbor_atoms, 2):
            atom1_label, vector1 = atom_combo[0]
            atom2_label, vector2 = atom_combo[1]
            triplet_sorted = tuple(sorted([atom1_label, central_label, atom2_label]))
            if triplet_sorted not in unique_triplets and atom1_label != atom2_label:
                # Calculate angle for unique triplet including the central atom
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
                
        # Form triplets involving only neighboring atoms (e.g., O-O-O around a central atom)
        for atom_combo in combinations(neighbor_atoms, 3):
            atom1_label, vector1 = atom_combo[0]
            atom2_label, vector2 = atom_combo[1]
            atom3_label, vector3 = atom_combo[2]
            triplet_sorted = tuple(sorted([atom1_label, atom2_label, atom3_label]))
            if triplet_sorted not in unique_triplets and len({atom1_label, atom2_label, atom3_label}) == 3:
                # Include only angles if all three atoms are oxygen and the angle is in the include range
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
                        continue
                else:
                    # Non-O-O-O angles are still added
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
    dihedral_quadruplets = []  # Store valid dihedral quadruplets
    unique_quadruplets = set()  # Track processed quadruplets

    # Iterate over all combinations of bond pairs
    for bond1, bond2 in combinations(bond_pairs, 2):
        # Ensure bonds share the same polyhedron (same central atom)
        if bond1['central_label'] != bond2['central_label']:
            continue  # Skip if bonds belong to different polyhedra

        # Extract polyhedral center
        polyhedron_center = bond1['central_label']

        # Extract atom labels
        atoms = {bond1['atom1_label'], bond1['atom2_label'], bond2['atom1_label'], bond2['atom2_label']}

        # Ensure we have exactly four unique atoms
        if len(atoms) != 4:
            continue  # Skip invalid quadruplets (e.g., repeated atoms)

        # Assign atoms correctly
        atom1, atom2, atom3, atom4 = atoms  # Assign sorted atom labels

        # Extract atom positions
        pos1 = np.array(bond1['atom1_position'])
        pos2 = np.array(bond1['atom2_position'])  # Shared central atom
        pos3 = np.array(bond2['atom1_position'])  # Another central atom
        pos4 = np.array(bond2['atom2_position'])

        # Ensure valid positions (no duplicate positions)
        unique_positions = {tuple(pos1), tuple(pos2), tuple(pos3), tuple(pos4)}
        if len(unique_positions) != 4:
            continue  # Skip quadruplets with duplicate positions

        # Compute vectors
        v1 = pos2 - pos1
        v2 = pos3 - pos2
        v3 = pos4 - pos3

        # Check for collinear or zero-length vectors
        if (
            np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6 or np.linalg.norm(v3) < 1e-6 or
            np.linalg.norm(np.cross(v1, v2)) < 1e-12
        ):
            continue  # Skip invalid quadruplets

        # Calculate dihedral angle
        dihedral_angle = calculate_dihedral(v1, v2, v3)
        if dihedral_angle is None:
            continue  # Skip invalid calculations

        # Convert to degrees for filtering
        dihedral_angle_degrees = np.degrees(dihedral_angle)

        # Exclude angles near 0° or 180°
        if abs(dihedral_angle_degrees) < angle_threshold or abs(dihedral_angle_degrees) > (180 - angle_threshold):
            #print(f"Excluded: {dihedral_angle_degrees:.2f}° (near 0° or 180°)")
            continue  # Skip nearly collinear angles

        # Store as an ordered tuple to avoid duplicates
        quadruplet = tuple(sorted([atom1, atom2, atom3, atom4]))

        if quadruplet in unique_quadruplets:
            continue  # Skip duplicates

        # Store valid quadruplet with dihedral angle
        dihedral_quadruplets.append({
            'quadruplet': quadruplet,
            'angle': dihedral_angle_degrees,  # Store in degrees
            'polyhedron_center': polyhedron_center
        })
        unique_quadruplets.add(quadruplet)

    return dihedral_quadruplets

#----------------------------------------------------------------------------
def detect_edge_bonds(bond_vectors, threshold=0.005):
    """
    Identify V-O bonds where at least one atom is very close to the unit cell boundary.

    Parameters:
    - bond_vectors: Dictionary containing bond data for 'Zr-O', 'V-O', and 'O-O'.
    - threshold: Float, distance from the unit cell edge (default: 0.005).

    Returns:
    - edge_bonds: List of V-O bond dictionaries where at least one atom is near the edge.
    """
    edge_bonds = []

    # Only process V-O bonds
    if 'V-O' in bond_vectors:
        for bond in bond_vectors['V-O']:  # ✅ Focus only on 'V-O' bonds
            pos1 = bond['atom1']['position']
            pos2 = bond['atom2']['position']

            # Check if either atom is close to 0 or 1
            if (
                any(coord < threshold or coord > (1 - threshold) for coord in pos1) or
                any(coord < threshold or coord > (1 - threshold) for coord in pos2)
            ):
                edge_bonds.append(bond)

    print(f"[INFO] Detected {len(edge_bonds)} V-O bonds near the unit cell edge.")
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

        # Variable name using the two atom labels to avoid naming conflicts
        var_name = f'bond_length_{atom1_label}_{atom2_label}_{phase}'

        # Bond expression based on coordinates of atom1 and atom2
        bond_expr = f"((x_{atom1_label}_{phase} - x_{atom2_label}_{phase})**2 + " \
                    f"(y_{atom1_label}_{phase} - y_{atom2_label}_{phase})**2 + " \
                    f"(z_{atom1_label}_{phase} - z_{atom2_label}_{phase})**2)**0.5"
        
        # Add bond constraint to the dictionary
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
        angle_name = angle['angle name']  # Directly use the angle name from the triplet
        angle_value = angle['angle']  # The calculated angle value from the triplet

        # Check if angle value is valid (not None or NaN)
        if angle_value is None or np.isnan(angle_value):
            continue  # Skip adding this angle if it's invalid

        # Use angle_name to construct the variable name, e.g., 'angle_O1_Zr_O2_Phase1'
        var_name = f'angle_{"_".join(angle_name)}_{phase}'
        
        # Construct the angle expression based on the atom labels in angle_name
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
        
        # Add to the constraint dictionary
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
        quadruplet = dihedral['quadruplet']  # Extract the quadruplet of atoms
        dihedral_value = dihedral['angle']  # Extract the dihedral angle value
        polyhedron_center = dihedral['polyhedron_center'] # Extract information about the centre of the polyhedron
        # Ensure valid dihedral angle
        if dihedral_value is None or np.isnan(dihedral_value):
            continue


        # Correct variable names
        atom1, atom2, atom3, atom4 = quadruplet  # Ensure correct unpacking

        # Create variable name for constraint
        var_name = f"dihedral_{'_'.join(quadruplet)}_{phase}"
        
        # Construct the dihedral angle expression
        dihedral_expr = (
            f"arctan(((x_{atom3}_{phase} - x_{atom2}_{phase}) * "
            f"(((y_{atom2}_{phase} - y_{atom1}_{phase}) * (z_{atom4}_{phase} - z_{atom3}_{phase})) - "
            f"((z_{atom2}_{phase} - z_{atom1}_{phase}) * (y_{atom4}_{phase} - y_{atom3}_{phase}))) + "
            f"((y_{atom3}_{phase} - y_{atom2}_{phase}) * "
            f"(((z_{atom2}_{phase} - z_{atom1}_{phase}) * (x_{atom4}_{phase} - x_{atom3}_{phase})) - "
            f"((x_{atom2}_{phase} - x_{atom1}_{phase}) * (z_{atom4}_{phase} - z_{atom3}_{phase}))) + "
            f"((z_{atom3}_{phase} - z_{atom2}_{phase}) * "
            f"(((x_{atom2}_{phase} - x_{atom1}_{phase}) * (y_{atom4}_{phase} - y_{atom3}_{phase})) - "
            f"((y_{atom2}_{phase} - y_{atom1}_{phase}) * (x_{atom4}_{phase} - x_{atom3}_{phase}))))) / "
            f"(((x_{atom2}_{phase} - x_{atom1}_{phase}) * (x_{atom3}_{phase} - x_{atom2}_{phase})) + "
            f"((y_{atom2}_{phase} - y_{atom1}_{phase}) * (y_{atom3}_{phase} - y_{atom2}_{phase})) + "
            f"((z_{atom2}_{phase} - z_{atom1}_{phase}) * (z_{atom3}_{phase} - z_{atom2}_{phase})))))"
        )

        # Store in constraint dictionary
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

        # Detect edge V-O bonds
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

            # Step 1: Identify shared oxygen atoms for V-O-V bonds
            shared_oxygen_dict = {}

            for var_name, data in constrain_dict_bonds.items():
                atom1, atom2 = data['atoms']

                if atom1.startswith('V') and atom2.startswith('O'):  # Ensure it's a V-O bond
                    oxygen_label = atom2

                    if oxygen_label not in shared_oxygen_dict:
                        shared_oxygen_dict[oxygen_label] = {
                            "bond_vars": [],
                            "expressions": []
                        }

                    shared_oxygen_dict[oxygen_label]["bond_vars"].append(var_name)
                    shared_oxygen_dict[oxygen_label]["expressions"].append(data['expression'])

            # Remove non-shared oxygens (must have exactly TWO V neighbors)
            shared_oxygen_dict = {k: v for k, v in shared_oxygen_dict.items() if len(v["bond_vars"]) == 2}

            # Step 2: Identify V-O bonds at the unit cell edge (structured like shared_oxygen_dict)
            edge_bond_dict = {}
            
            for bond in edge_bonds:
                atom1_label = bond["atom1"]["label"]  # V
                atom2_label = bond["atom2"]["label"]  # O
            
                # Store bonds only under the oxygen label
                if atom2_label not in edge_bond_dict:
                    edge_bond_dict[atom2_label] = []
            
                bond_var_name = f"bond_length_{atom1_label}_{atom2_label}_{phase_key}"
                edge_bond_dict[atom2_label].append(bond_var_name)
            
            print(f"[INFO] Identified {len(edge_bond_dict)} V-O edge bonds.")
            
            
            # Step 3: Identify problematic bonds (length < 1.6 Å or > 1.9 Å)
            # Calculate current bond vectors from the current structure
            bond_vectors_current = get_polyhedral_bond_vectors(getattr(cpdf, phase).phase)
            
            
            problematic_bonds_dict = {}
            for bond_type in ['V-O']:  # You can extend this list to other bond types if needed
                for bond in bond_vectors_current[bond_type]:
                    if bond['length'] < 1.6 or bond['length'] > 1.9:
                        atom1_label = bond["atom1"]["label"]
                        atom2_label = bond["atom2"]["label"]
                        bond_key = f"bond_length_{atom1_label}_{atom2_label}_{phase_key}"
                        problematic_bonds_dict[bond_key] = bond  # store the problematic bond information
            
            print(f"[INFO] Detected {len(problematic_bonds_dict)} problematic V-O bonds (length <1.6 or >1.9 Å).")
            

            # Step 4: Apply constraints
            # --------------------------------------------------------------------
            
            for var_name, data in constrain_dict_bonds.items():
                try:
                    fit.newVar(var_name, tags=['bond_length'])
                    fit.constrain(var_name, data['expression'])
            
                    # V-O-V shared bonds (existing logic)
                    if any(var_name in info["bond_vars"] for info in shared_oxygen_dict.values()):
                        strict_sigma = 1e-8
                        lb = 0.95 * data['relative_length']
                        ub = 1.05 * data['relative_length']
                        print(f"[V-O-V BOND] {var_name}: lb={lb:.6f}, ub={ub:.6f}, sigma={strict_sigma}")
            
                    # Edge V-O bonds (existing logic)
                    elif any(var_name in bond_list for bond_list in edge_bond_dict.values()):
                        strict_sigma = 1e-7
                        lb = 0.95 * data['relative_length']
                        ub = 1.05 * data['relative_length']
                        print(f"[EDGE BOND] {var_name}: lb={lb:.6f}, ub={ub:.6f}, sigma={strict_sigma}")
            
                    # NEW: Problematic bonds (length < 1.6 Å or > 1.9 Å)
                    elif var_name in problematic_bonds_dict:
                        # Extract initial relative length from the reference bond_vectors
                        initial_rel_length = data['relative_length']
                        strict_sigma = 1e-7
                        lb = 0.95 * initial_rel_length
                        ub = 1.05 * initial_rel_length
                        print(f"[PROBLEMATIC BOND] {var_name}: lb={lb:.6f}, ub={ub:.6f}, sigma={strict_sigma}")
            
                    # Normal bonds (existing logic)
                    else:
                        strict_sigma = constrain_bonds[1]
                        lb = 0.95 * data['relative_length']
                        ub = 1.05 * data['relative_length']
                        print(f"[NORMAL BOND] {var_name}: lb={lb:.6f}, ub={ub:.6f}, sigma={strict_sigma}")
            
                    fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=strict_sigma)
            
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
                    fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=constrain_angles[1])

                    print(f"[INFO] {var_name}: dihedral={data['angle']:.2f}° (±{ang_limit}°)")
                except Exception as e:
                    print(f"[WARNING] Skipped dihedral {var_name} due to error: {e}")

        # --------------------------------------------------------------------
        # Cleanup: Remove variables that are None
        # --------------------------------------------------------------------
        for name in dir(fit):
            if name.startswith(('bond_', 'angle_', 'dihedral_')):
                if getattr(fit, name).value is None:
                    fit.unconstrain(getattr(fit, name))
                    fit.delVar(getattr(fit, name))
                    print(f"[INFO] Removed '{name}' due to a missing or incorrect value.")

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
        # Example parameter name: 'x_0', 'y_3', 'Uiso_2', etc.
        parname = par.name
        parts = parname.split("_")
        coord = parts[0]        # 'x', 'y', 'z', or 'Uiso', 'U11', etc.
        idx = int(parts[1])     # scatterer index

        # Retrieve scatterer and atom name
        scatterer = sgpar.scatterers[idx]
        atom_name = scatterer.name  # e.g. 'O1', 'Zr'
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
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print(f"Fitting stage {i}")
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    cpdf.setCalculationRange(fitting_range[0],fitting_range[1],myrstep)
    cpdf.setResidualEquation(residualEquation)
    fit.fix('all')
    for step in fitting_order:
        try:
            print(f'Freeing paramater: {step}')
            #fit.fix('all')
            fit.free(step)
            #leastsq(fit.residual, fit.values, full_output=True, **convergence_options)
            #least_squares(fit.residual, fit.values, method = 'trf', **convergence_options)
            optimizer = 'L-BFGS-B'
            minimize(fit.scalarResidual, fit.values, method=optimizer, options = convergence_options)
        except:
            continue
     
    
    res = saveResults(i,fit, output_results_path)

    #Statistics on the bond lengths and angle distributionsc
    for phase_name in cpdf._generators:
        phase = getattr(cpdf,str(phase_name)).phase
        visuals = visualize_fit_summary(fit, phase, output_results_path+str(i)+'_'+str(phase_name)+'_', font_size=14, label_font_size=20)
    return

#-----------------------------------------------------------------------------
def refinement_basic(cpdf, anisotropic, unified_Uiso, sgoffset = [0,0,0]):
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
    # setup fit contribution for PDF
    # create FitRecipe to manage the refinement
    fit = FitRecipe()
    fit.addContribution(cpdf)
    
    
    #------------------ 0a ------------------#
# #   add Q-broad as a fitting paramater (optional)
#     qbroad_ini = fit.newVar('qbroad', value=qbroad, tags=['qbroad'])
#     for i, phase in enumerate(cpdf._generators):
#         fit.constrain(getattr(cpdf, phase).qbroad, qbroad_ini)
#         fit.restrain(getattr(cpdf, phase).qbroad, lb=0.0, sig=0.0001)  

    #------------------ 0b ------------------#
# #   add Q-damp as a fitting paramater (optional)
#     qdamp_ini = fit.newVar('qdamp', value=qdamp, tags=['qdamp'])
#     for i, phase in enumerate(cpdf._generators):
#         fit.constrain(getattr(cpdf, phase).qdamp, qdamp_ini)
#         fit.restrain(getattr(cpdf, phase).qdamp, lb=0.0, sig=0.0001)              
    
    #------------------ 1 ------------------# 
    #Generate a phase equation. It pre-defines scale factors s* parameters for each phase.
    # It has to be done before point (3)
    phase_equation = ''
    for i, phase in enumerate(cpdf._generators):
        phase_equation += 's_' + str(phase) + '*' + str(phase) + ' + '
    cpdf.setEquation(phase_equation[:-3])
    print('equation:', cpdf.getEquation())
    
    

    #------------------ 2a ------------------#

    #add delta1 as a shared fittiing paramater common for all phases
    # delta = fit.newVar('delta1', value=0.1, tags=['delta1'])
    # for i, phase in enumerate(cpdf._generators):
    #     fit.constrain(getattr(cpdf, phase).delta1, delta)
    #     fit.restrain(getattr(cpdf, phase).delta1, lb=0.0, sig=0.0001)
    
    #add delta1 as a fittiing paramater for each phase independently
    # for i, phase in enumerate(cpdf._generators):
    #     fit.addVar(getattr(cpdf, phase).delta1, name='delta1_'+str(phase), value=0.05, tags=['delta1', str(phase), 'delta1_'+str(phase), 'delta'])
    #     fit.restrain(getattr(cpdf, phase).delta1, lb=0.0, ub = 2, sig=0.00001)  


    #------------------ 2b ------------------#
    #add delta2 as a shared fittiing paramater common for all phases
    # delta = fit.newVar('delta2', value=2, tags=['delta2'])
    # for i, phase in enumerate(cpdf._generators):
    #     fit.constrain(getattr(cpdf, phase).delta2, delta)
    #     fit.restrain(getattr(cpdf, phase).delta2, lb=0.0, sig=0.0001)
    
    #add delta2 as a fittiing paramater for each phase independently
    for i, phase in enumerate(cpdf._generators):
        fit.addVar(getattr(cpdf, phase).delta2, name='delta2_'+str(phase), value=2.0, tags=['delta2', str(phase), 'delta2_'+str(phase), 'delta'])
        fit.restrain(getattr(cpdf, phase).delta2, lb=0.0, ub = 5, scaled = True, sig=0.005)   


    #------------------ 3 ------------------#

    #add other fittiing paramater for each phase
    for i, phase in enumerate(cpdf._generators):

    

        #------------------ 4 ------------------#    
    #add scale factors s* 
        fit.addVar(getattr(cpdf,'s_'+str(phase)), value = 0.1, tags = ['scale', str(phase),'s_'+str(phase)])
        fit.restrain(getattr(cpdf, 's_'+str(phase)), lb=0.0, scaled = True, sig = 0.0005)
        

        #------------------ 5 ------------------#    

        #determine independent parameters based on the space group
        spaceGroup = str(list(ciffile.values())[i][0])
        sgpar = constrainAsSpaceGroup(getattr(cpdf, phase).phase, spaceGroup, sgoffset = sgoffset)
        #print("Space group parameters are:", ', '.join(p.name for p in sgpar))
    

        #------------------ 6 ------------------#

        #add lattice paramaters    
        for par in sgpar.latpars:
            fit.addVar(par, value = par.value, name = par.name + '_' + str(phase), fixed = False, tags=['lat', str(phase), 'lat_' + str(phase)])
        
  

        #------------------ 7a ------------------#         

        #atomic displacement paramaters ADPs
        
        #anisotropic ADPs are added as fitting paramaters according to symmetry restrictions    
        if anisotropic == True:
            (getattr(cpdf, phase)).stru.anisotropy = True
            print('Adding anisotropic displacement paramaters.')

            for par in sgpar.adppars:
                atom_label = par.par.obj.label
                atom_symbol = par.par.obj.element
                if atom_symbol == 'V':
                    value = 0.01
                if atom_symbol == 'O':
                    value = 0.025
                if atom_symbol == 'Zr':
                    value = 0.0065

            #tag ADPss with element symbols
                name=par.name+'_'+atom_label+'_'+str(phase)
                tags = ['adp', 'adp_' + atom_label, 'adp_' + atom_symbol + '_' + str(phase),'adp_'+str(phase), 'adp_' + atom_symbol, str(phase)]
                fit.addVar(par, value=value, name=name, tags=tags)
                #fit.addVar(par, value = 0.005, name = par.name + '_' + str(phase), fixed = False, tags = ['adp', str(phase), 'adp_' + str(phase)])
                fit.restrain(par, lb=0.0, ub=0.1, scaled = True, sig=0.0005)

        #------------------ 7b ------------------#         
        #Allow isotropic displacement parameters  
        else:
        # sgpars.adppars have some funky labeling convention. We will relabel the atom names to match those in the rest of this script. 
    
            mapped_adppars =  map_sgpar_params(sgpar, 'adppars')
            added_adps = set()
            for par in sgpar.adppars:
                try:
                    atom_symbol = par.par.obj.element
                    parameter_name = par.name
                    atom_label = mapped_adppars[parameter_name][1]
                    if atom_label not in added_adps:
                        added_adps.add(atom_label)
                except:
                    pass
            if unified_Uiso ==True:
#shared Uiso values for atoms of the same element 
                print('Adding isotropic displacement parameters as unified values.')
                (getattr(cpdf, phase)).stru.anisotropy = False
        
        #initial guesses of the Uiso values
                Uiso_O = fit.newVar('Uiso_O_'+str(phase), value=0.025, tags=['adp', str(phase), 'adp_' + str(phase), 'adp_O'])
                Uiso_Zr = fit.newVar('Uiso_Zr_'+str(phase), value=0.0065,tags=['adp', str(phase), 'adp_' + str(phase), 'adp_Zr'])
                Uiso_V = fit.newVar('Uiso_V_'+str(phase), value=0.01, tags=['adp', str(phase), 'adp_' + str(phase), 'adp_V'])
                
                for atom in getattr(cpdf, phase).phase.getScatterers():
                    if atom.element=='O':
                        fit.constrain(atom.Uiso, Uiso_O)
                        fit.restrain(atom.Uiso, lb=0.0, ub = 0.1, scaled = True, sig=0.0005)  
    
                     
                    if atom.element=='Zr':
                        fit.constrain(atom.Uiso, Uiso_Zr)
                        fit.restrain(atom.Uiso, lb=0.0, ub = 0.1, scaled = True, sig=0.0005)
    
                      
                    if atom.element=='V':
                        fit.constrain(atom.Uiso, Uiso_V)  
                        fit.restrain(atom.Uiso, lb=0.0, ub = 0.1, scaled = True, sig=0.0005)
            
            else:
                #all Uiso values are independent for all the atoms
                print('Adding isotropic displacement parameters as independent values.')
                (getattr(cpdf, phase)).stru.anisotropy = False

                for atom in getattr(cpdf, phase).phase.getScatterers():
                    if atom.name.upper() in added_adps: #added_adps set contains atom names in upper case
                #initial guesses of the Uiso values 
                        if atom.element == 'V':
                            value = 0.01
                        if atom.element == 'O':
                            value = 0.025
                        if atom.element == 'Zr':
                            value = 0.0065
                        fit.addVar(atom.Uiso, value = value, name = atom.name + '_' + str(phase), fixed = False, tags = ['adp', str(phase), 'adp_' + str(phase),'adp_'+str(atom.element)])
                        fit.restrain(atom.Uiso, lb=0.0, ub = 0.1, scaled = True, sig=0.0005)



        #------------------ 8 ------------------# 

# atom positions XYZ   
# atoms are labeled according to tags e.g.
# ['xyz', 'xyz_O1', 'xyz_O_Phase1', 'xyz_Phase1', 'xyz_O']
# so that specific atoms in a specific phase can be un-/fixed
        # Initialize the added_params for each phase
        added_params[str(phase)] = set()  # Initialize for each phase

# sgpars.xyzpars have some funky labeling convention. We will relabel the atom names to match those in the rest of this script.    
        mapped_xyzpars = map_sgpar_params(sgpar, 'xyzpars')
        for par in sgpar.xyzpars:
            try:
                atom_symbol = par.par.obj.element
                parameter_name = par.name
                
                #tag XYZs with element symbols
                mapped_name = mapped_xyzpars[parameter_name][0]
                atom_label = mapped_xyzpars[parameter_name][1]
                name_long = mapped_name+'_'+str(phase)
    
                tags = ['xyz','xyz_' + atom_symbol, 'xyz_' + atom_symbol + '_' + str(phase),'xyz_'+str(phase), str(phase)]

                fit.addVar(par, name=name_long, tags=tags)
                #fit.restrain(name_long, lb = par.par.value - pos_restrain, ub = par.par.value + pos_restrain, scaled = True, sig = 0.001)
                
                # Add the atom_label to the phase-specific added_params set
                added_params[str(phase)].add(atom_label)
                print(f"Constrained {name_long} at {par.par.value}: atom position added to variables.")
            except:
                pass   


            

        #------------------ 9 ------------------#

#add oxygen occupancy to allow vacancies
#Do not use if not needed
        # print('Adding oxygen occupancy.')
        # #initial guesses of the Uiso values
        # occ_O = fit.newVar('occ_O_'+str(phase), value=1.0, tags=['occ', str(phase), 'occ_' + str(phase), 'occ_O'])
        # for atom in getattr(cpdf, phase).phase.getScatterers():
        #     if atom.element=='O':
        #         fit.constrain(atom.occ, occ_O)
        #         fit.restrain(atom.occ, lb=0.0, ub = 1.0, sig=0.01)  
    

        #------------------ 10 ------------------#

#allow all phases to be potentially nano-sized

#add a sperical envelope functions for all the phases
    for i, phase in enumerate(cpdf._generators):   
        if i >= 0:           
            cpdf.registerFunction(sphericalCF, name='sphere_'+str(phase), argnames=['r', 'psize_'+str(phase)])
        else:
            pass
    #new phase equation
    phase_equation = ''
    for i, phase in enumerate(cpdf._generators):
        if i >= 0:
            phase_equation += 's_' + str(phase) + '*' + str(phase) + '*' + 'sphere_'+str(phase)  + ' + '
        else:
            pass
    cpdf.setEquation(phase_equation[:-3])
    print('equation:', cpdf.getEquation())
    
    for i, phase in enumerate(cpdf._generators):
        if i >= 0:
            fit.addVar(getattr(cpdf, 'psize_'+str(phase)), value = 100.0, fixed = False, tags = ['psize', 'psize_'+str(phase), str(phase)] )
            fit.restrain(getattr(cpdf, 'psize_'+str(phase)), lb=0.0, scaled = True, sig = 0.1)
        else:
            pass
#show diagnostic information
    fit.fithooks[0].verbose = 2
  

        #------------------ 11 ------------------#
   
#calculate bond vectors from all the phases and store as a global immutable variable
    global global_bond_vectors
    for phase in cpdf._generators:
        print(f"Calculating bond vectors for {phase}")
        bond_vectors = get_polyhedral_bond_vectors(getattr(cpdf, phase).phase)
        global_bond_vectors[phase] = bond_vectors

    return fit
#-----------------------------------------------------------------------------

def modify_fit(fit, spacegroup, sgoffset = [0,0,0]):
    """
    Modify the fitting recipe by updating the space group and regenerating atomic position variables,
    constraints, and restraints. This function adds new atom positions from a lower symmetry strucure,
    but retains original lattice constants

    Parameters:
    - fit: FitRecipe object, the current fitting recipe containing refinement variables and constraints.
    - spacegroup: String, the new space group to apply to the structure.
    - sgoffset: List of three floats, offset applied to the origin of the space group symmetry operations
      (default: [0, 0, 0]).


    Returns:
    - fit: Updated FitRecipe object with the new space group constraints and variables.
    """
    global added_params
    
    
    # Step 1: Read out all atom position variables from the fit
    old_xyz_vars = {}  # Temporary dictionary to store current variable names and values
    
    for name in fit.names:
        if name.startswith('x_') or name.startswith('y_') or name.startswith('z_'):
            var_value = getattr(fit, name).value
            old_xyz_vars[name] = var_value  
    
    # Step 3: Delete the old variables and clear added_params for each phase
    for name in old_xyz_vars.keys():
        fit.unconstrain(getattr(fit, name))
        fit.delVar(getattr(fit, name))
        print(f"{name}: old variable deleted")
    
    # Clear added_params for each phase
    for phase in fit.cpdf._generators:
        added_params[str(phase)] = set()
    
    # Step 4: remove rigid body restrictions
    # This is important when the symmetry is changed, espcially for stching from low to high
    for name in dir(fit):
        if name.startswith('bond_') or name.startswith('angle_') or name.startswith('dihedral_'):
            fit.unconstrain(getattr(fit,name))
            fit.delVar(getattr(fit, name))
            print(f"{name}: old variable deleted")


    # Step 4: Apply the new space group and generate new variables
    for phase in fit.cpdf._generators:
        
        try:
            # Apply new space group constraints
            sgpar = constrainAsSpaceGroup(getattr(fit.cpdf, phase).phase, spacegroup)
            
            #SUPER IMPORTANT!!!
            # Ensure that existing constraints are cleared
            sgpar._clearConstraints()
        except Exception as e:
            print(f"Error in applying space group: {e}")
            pass
        
        # Step 5: Map the new variables to actual atom names
        mapped_xyzpars = map_sgpar_params(sgpar, 'xyzpars')

 #Now, iterate over the new variables and apply the values from the old dictionary
        for par in sgpar.xyzpars:
            try:
                atom_symbol = par.par.obj.element
                parameter_name = par.name
                # Map the new variable to the old variable name (reverse mapping)
                mapped_name = mapped_xyzpars[parameter_name][0]
                atom_label = mapped_xyzpars[parameter_name][1]
                name_long = mapped_name + '_' + str(phase)
                
                tags = ['xyz', 'xyz_' + atom_symbol, 'xyz_' + atom_symbol + '_' + str(phase), 'xyz_' + str(phase), str(phase)]

                # Get the old variable value if it exists, or use the current value
                if name_long in old_xyz_vars:
                    old_value = old_xyz_vars[name_long]
                else:
                    old_value = par.par.value  # Fallback to the current value if not found in the old vars


                # Add the variable to the fit and apply the value
                fit.addVar(par, value = old_value, name=name_long, tags=tags)
                # if atom_symbol == 'Zr':
                #     pos_restrain = 0.01
                #     fit.restrain(name_long, lb=old_value - pos_restrain, ub=old_value + pos_restrain, scaled = True, sig=0.001)
                #     print(f"Position of {name_long} restrained")
                # Add the atom_label to the phase-specific added_params set
                added_params[str(phase)].add(atom_label)

                print(f"Constrained {name_long} at {old_value}: atom position added to variables.")
                #print(f"Default value: {par.par.value}. Temporary refined value: {old_value}")
            except Exception as e:
                print(f"Error processing {parameter_name}: {e}")
                
    # Step 6: Enforce Pseudo-Cubic Constraints for Lattice Parameters and anisotropic ADPs
    old_lattice_vars = {}  # Temporary dictionary to store current variable names and values
    
    for name in fit.names:
        if name.startswith('a_') or name.startswith('b_') or name.startswith('c_') or name.startswith('alpha_') or name.startswith('beta_') or name.startswith('gamma_'):
            var_value = getattr(fit, name).value
            old_lattice_vars[name] = var_value  
    
    for name in old_lattice_vars.keys():
        fit.unconstrain(getattr(fit, name))
        fit.delVar(getattr(fit, name))
        print(f"{name}: old variable deleted")
    
    for phase in fit.cpdf._generators:
        spaceGroup = str(list(ciffile.values())[0][0])
        sgpar = constrainAsSpaceGroup(getattr(cpdf, phase).phase, spaceGroup, sgoffset = sgoffset)
        for par in sgpar.latpars:
            name = par.name + '_' + str(phase)
            old_value = old_lattice_vars[name]
            fit.addVar(par, value = old_value, name = name, fixed = False, tags=['lat', str(phase), 'lat_' + str(phase)])
            print(f"Constrained {name} at {old_value}.")
        
        
    return fit
#-----------------------------------------------------------------------------





# =============================================================================
#                              PROJECT SETUP
# =============================================================================

# =============================== Input Definitions ===========================
# Define project name and directories
project_name = 'ZirconiumVanadate75Cperiodic/'
xrd_directory = 'data/'  # Directory containing diffraction data
cif_directory = 'CIFs/'  # Directory containing CIF files
fit_directory = 'fits/'  # Base directory for storing refinement results

# =============================== XRD Data ====================================
# Specify diffraction data file
mypowderdata = 'PDF_ZrV2O7_061_75C_avg_126_145_00000.dat'

# =============================== Output Setup ================================
# setup date and time format strings for naming of the fit results
# DDMMYY_hms
time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
# Create output directory for results
output_results_path = fit_directory + project_name + str(time_stamp) + '/'
if not os.path.isdir(output_results_path):
    os.makedirs(output_results_path)

# =============================== Phase Setup ================================
# Define structural phases
#ciffile = {'98-005-9396_ZrV2O7.cif': ['Pa-3', False, (1, 1, 1)], '98-005-9396_ZrV2O7_2.cif': ['Pa-3', False, (1, 1, 1)]}
ciffile = {'98-005-9396_ZrV2O7.cif': ['Pa-3', True, (1, 1, 1)]}
#ciffile = {'Phase0_6.cif': ['Pa-3', True, (1, 1, 1)]}


# ========================== Atomic Composition ===============================
# Define atomic composition
composition = 'O7 V2 Zr1'

# =========================== Instrumental Parameters =========================
qdamp = 2.70577268e-02  # Instrumental broadening
qbroad = 2.40376789e-06  # Sample-induced broadening

# ================= Atomic Displacement Parameters (ADPs) ====================
# Decide how to handle ADPs: anisotropic or isotropic
anisotropic = False  # Use isotropic ADPs
unified_Uiso = True  # Unify Uiso for atoms of the same element

# ======================= Space Group Offset ================================
sgoffset = [0.0, 0.0, 0.0]  # Offset applied to symmetry operations

# ======================= PDF Generation Parameters ==========================
myrange = (0.0, 80)  # Range of r values for PDF
myrstep = 0.05       # Step size for r values

# ================== Fitting Procedure Parameters ============================
#convergence_options = {'ftol': 1e-4, 'gtol': 1e-6, 'xtol': 1e-5,'disp': True}
convergence_options = {'disp': True}
#relevant only for some fitting methods

# =============================================================================
#                             INITIALIZATION
# =============================================================================
# Calculate available CPU cores for processing
syst_cores = multiprocessing.cpu_count()
cpu_percent = psutil.cpu_percent()
avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
ncpu = int(np.max([1, avail_cores]))  # Ensure at least one core is used
pool = Pool(processes=ncpu)

# Generate the initial PDF
r0, g0, cfg = generatePDF(
    xrd_directory + mypowderdata,
    composition,
    qmin=0.0,
    qmax=22,
    myrange=myrange,
    myrstep=myrstep
)
cpdf = phase(r0, g0, cfg, cif_directory, ciffile, myrange, myrstep, qdamp, qbroad)

#global variables to keep track of the data
# Global dictionary to keep track of added XYZ positions in the refinement recipe for each phase
added_params = {}

#bond vectors to be calculated for each phase
global_bond_vectors = {}


# Initialize the fit object with default constraints
fit0 = refinement_basic(
    cpdf,
    anisotropic=anisotropic,
    unified_Uiso=unified_Uiso,
    sgoffset=sgoffset
)


# =============================================================================
#                               FITTING STEPS
# =============================================================================

## ========================== Step 0: Initial Fit =============================
i = 0
fitting_order = ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz']
fitting_range = [1.5, 27]
residualEquation = 'resv'
constrain_bonds = (True, 0.001)
constrain_angles = (True, 0.001)
constrain_dihedrals = (False, 0.001)
fit0 = modify_fit(fit0, 'Pa-3', sgoffset=sgoffset)
fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path, **convergence_options)


# ========================== Step 1: Refinement ==============================
i = 1
constrain_bonds = (True, 0.0001)
constrain_angles = (True, 0.0001)

fit0 = modify_fit(fit0, 'Pa-3', sgoffset=sgoffset)
fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path, **convergence_options)

# ========================== Step 2: Adjust Symmetry =========================
i = 2
constrain_bonds = (True, 0.001)
constrain_angles = (True, 0.001)

fit0 = modify_fit(fit0, 'P213', sgoffset=sgoffset)
fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path, **convergence_options)
# ========================== Step 3: Adjust Symmetry =========================
i = 3
fit0 = modify_fit(fit0, 'P23', sgoffset=sgoffset)
constrain_bonds = (True, 0.001)
constrain_angles = (True, 0.001)

fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path, **convergence_options)
# ========================== Step 4: Further Refinement ======================
i = 4
constrain_bonds = (True, 0.0001)
constrain_angles = (True, 0.0001)

fit0 = modify_fit(fit0, 'P23', sgoffset=sgoffset)
fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path)

# ========================== Step 5: Lowest Symmetry =========================
i = 5
fit0 = modify_fit(fit0, 'P1', sgoffset=sgoffset)
constrain_bonds = (True, 0.001)
constrain_angles = (True, 0.001)

fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path, **convergence_options)

# ========================== Step 6: Lowest Symmetry =========================
i = 6
fit0 = modify_fit(fit0, 'P1', sgoffset=sgoffset)
constrain_bonds = (True, 0.0001)
constrain_angles = (True, 0.0001)

fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path, **convergence_options)

# ========================== Step 7: Extend Range =========================
# i = 7
# fitting_range = [1.5, 35]
# fit0 = modify_fit(fit0, 'P1', sgoffset=sgoffset)
# constrain_bonds = (True, 0.0001)
# constrain_angles = (True, 0.0001)

fit0 = refinement_RigidBody(fit0, cpdf, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False)
fit_me(i, fitting_range, myrstep, fitting_order, fit0, cpdf, residualEquation, output_results_path, **convergence_options)
# =============================================================================
#                             FINALIZE RESULTS
# =============================================================================
finalize_results(cpdf, fit0, output_results_path, myrange, myrstep)

# #==============================================================================
# # Simulate PDFs
# # =============================================================================
# # file = cif_directory + list(ciffile.keys())[0]
# # df_sim = simulatePDF(file, rmin=myrange[0], rmax=myrange[1], rstep = myrstep, qmin = cfg.qmin, qmax = cfg.qmax, qdamp=qdamp, qbroad=qbroad, scale = 5e-01, delta2 = 2)
# # df_sim.to_csv('output.csv', index = None)
# # plt.plot(df_sim['r'], df_sim['g'])
