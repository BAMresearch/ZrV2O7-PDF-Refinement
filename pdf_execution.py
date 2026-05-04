"""
================================================================================
  Automated Structural Refinement via PDF Analysis
================================================================================

Author: Tomasz Stawski (tomasz.stawski@bam.de, tomasz.stawski@gmail.com)
Testing, Implementation and Analyses: Aiste Miliute (aiste.miliute@bam.de)
Version: 1.3.0
License: MIT License

DESCRIPTION:
This script serves as the main execution environment for performing automated,
multi-stage structural refinements of crystalline materials against X-ray
scattering data. It leverages an object-oriented framework defined in the
accompanying 'pdf_refinement_core.py' module.

The workflow is organised into several distinct stages:

1.  **Configuration**: All experimental parameters, file paths, and model
    details are defined in a centralised dictionary ('project_config').

2.  **Initialisation**: The script instantiates the necessary controller and
    manager classes from the imported module, configuring the environment for
    parallel processing to optimise computational performance.

3.  **Data Processing**: The raw experimental scattering data is processed to
    generate the Pair Distribution Function (PDF), G(r), which forms the basis
    for the refinement.

4.  **Model Construction**: A theoretical PDF is calculated from an initial
    structural model provided in a CIF. This establishes the initial FitRecipe 
    for the overall refinement.

5.  **Sequential Refinement**: The core of the script executes a series of
    refinement steps. The theoretical model is iteratively fitted to the
    experimental PDF by adjusting structural parameters (e.g., lattice
    constants, atomic positions, ADPs). The workflow is designed to 
    systematically explore different structural symmetries and apply 
    chemically motivated rigid-body constraints.
"""

# =============================================================================
# 1. CORE PROJECT CONFIGURATION
# =============================================================================
# This dictionary ('project_config') holds the global settings and structural 
# definitions required by the underlying 'pdf_refinement_core' classes.
# It supports single or multiple datasets via the 'dataset_list' array.

project_config = {
    # Project and Directory Management
    'project_name': 'YourProjectName/', # Root name for output folders
    'xrd_directory': 'dataXRD/',  # Location of scattering data
    'data_dir_gr': 'dataXRDandGofRs/',  # Location of precalculated .gr files (Fallback)
    'cif_directory': 'CIFs/',     # Location of structural models (.cif)
    'fit_directory': 'fits/',     # Default output for refinement results
    
    # Dataset Selection
    # Specify active sequences within the 'dataset_list'.
    'dataset_list': [
        'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat',
        'PDF_ZrV2O7_061_60C_avg_66_85_00000.dat',
        'PDF_ZrV2O7_061_70C_avg_106_125_00000.dat',
        'PDF_ZrV2O7_061_75C_avg_126_145_00000.dat',
        'PDF_ZrV2O7_061_80C_avg_146_165_00000.dat',
        'PDF_ZrV2O7_061_85C_avg_166_185_00000.dat',
        'PDF_ZrV2O7_061_90C_avg_186_205_00000.dat',
        'PDF_ZrV2O7_061_95C_avg_206_225_00000.dat',
        'PDF_ZrV2O7_061_100C_avg_226_245_00000.dat',
        'PDF_ZrV2O7_061_105C_avg_246_265_00000.dat',
        'PDF_ZrV2O7_061_111C_avg_266_285_00000.dat',
        'PDF_ZrV2O7_061_117C_avg_286_305_00000.dat',
        'PDF_ZrV2O7_061_122C_avg_306_325_00000.dat',
        'PDF_ZrV2O7_061_154C_avg_326_345_00000.dat',
        'PDF_ZrV2O7_061_209C_avg_346_365_00000.dat',
        'PDF_ZrV2O7_061_264C_avg_366_385_00000.dat',
        'PDF_ZrV2O7_061_318C_avg_386_405_00000.dat',
        'PDF_ZrV2O7_061_372C_avg_426_445_00000.dat',
        'PDF_ZrV2O7_061_427C_avg_446_465_00000.dat',
        'PDF_ZrV2O7_061_481C_avg_466_485_00000.dat',
        'PDF_ZrV2O7_061_536C_avg_486_505_00000.dat',
        'PDF_ZrV2O7_061_590C_avg_506_525_00000.dat',
        'PDF_ZrV2O7_061_644C_avg_526_545_00000.dat',
        'PDF_ZrV2O7_061_650C_avg_546_565_00000.dat',
        'PDF_ZrV2O7_061_655C_avg_566_585_00000.dat',
        'PDF_ZrV2O7_061_661C_avg_586_605_00000.dat',
        'PDF_ZrV2O7_061_666C_avg_606_625_00000.dat',
        'PDF_ZrV2O7_061_672C_avg_626_645_00000.dat',
        'PDF_ZrV2O7_061_677C_avg_646_665_00000.dat',
        'PDF_ZrV2O7_061_688C_avg_666_685_00000.dat',
        'PDF_ZrV2O7_061_699C_avg_686_705_00000.dat',
        # 'PDF_ZrV2O7_061_209C_avg_346_745_00000.dat',
        # 'PDF_ZrV2O7_061_209C_avg_726_745_00000.dat'
    ],
    
    # Structural and Chemical Definitions
    # 'ciffile' dictionary defines the structural starting point.
    # Key: Filename of the starting CIF model located in cif_directory.
    # Value (list containing 3 elements):
    #   1. String: Initial Space Group setting (e.g., 'Pa-3').
    #   2. Boolean: True enables periodic boundary lattice expansion; False isolates the structure into a finite nanoparticle.
    #   3. Tuple: Structural supercell definition, setting the multiplier across the (a, b, c) directions.
    'ciffile': {'98-005-9396_ZrV2O7.cif': ['Pa-3', True, (1, 1, 1)]}, 
    
    # 'composition' explicitly defines the overall empirical stoichiometry.
    # This actively guides atomic density estimations and normalisation routines during PDF baseline subtractions.
    'composition': 'O7 V2 Zr1',
    
    # Detailed Atom-Specific Settings
    # This dictionary independently configures initial parameters and constraints for distinct atomic species:
    #   - 'symbol': The strict elemental identifier, dictating the scattering factors loaded from internal tables.
    #   - 'Uiso': The initial baseline isotropic thermal displacement value (Å²) established prior to structural refinement.
    #   - 'polyhedron_center' / 'polyhedron_vertex': Dictates rigid-body coordination environments if specific geometrical symmetry constraints necessitate them.
    #   - 'cutoff': Tuple designating the minimum and maximum physically realistic bond lengths explicitly used by the forensics sub-module to detect mathematical anomalies.
    'detailed_composition': {
        'Zr': {'symbol': 'Zr', 'Uiso': 0.0065, 'polyhedron_center': True,  'polyhedron_vertex': False, 'cutoff': (1.8, 2.2)},
        'V':  {'symbol': 'V',  'Uiso': 0.0100, 'polyhedron_center': True,  'polyhedron_vertex': False, 'cutoff': (1.5, 2.4)},
        'O':  {'symbol': 'O',  'Uiso': 0.0250, 'polyhedron_center': False, 'polyhedron_vertex': True},
    },
    
    # Instrument and Resolution Parameters
    #   - 'qdamp': Resolution dampening factor correlated to the instrument profile. Dictates the exponential fall-off of the G(r) envelope at high r.
    #   - 'qbroad': Broadening parameter functionally widening structural peaks globally, correlated to specific instrumental delta-Q distributions.
    #   - 'qmax': The theoretical upper limit of the total momentum transfer Q (in Å⁻¹) employed during the Fourier transform, governing the inherent spatial resolution.
    'qdamp':  2.70577268e-02, 
    'qbroad': 2.40376789e-06, 
    'qmax':   22.0,           
    
    # Refinement Controls
    # Boolean directives actively dictating if instrumental baseline factors persistently remain fixed or if they are conditionally relaxed and freely co-optimised.
    'refine_qdamp':  False, 
    'refine_qbroad': False, 
    
    # Atomic Displacement Parameter (ADP) Handling Strategy
    # Defines the applied model for refining atomic thermal vibrations during the sequence.
    # Supported options:
    #   'isotropic'   : Assumes spherical thermal vibrations for all structural atoms. Only Uiso is refined.
    #   'anisotropic' : Employs a full tensor refinement, allowing for ellipsoidal thermal vibrations.
    #   'fixed_shape' : Applies a scaled anisotropic approach preserving the shape tensor from the input CIF.
    'adp_mode': 'isotropic',
    'sgoffset': [0.0, 0.0, 0.0], # Space group origin offset, if applicable
    
    # Calculation Range and Step Size (Å)
    'myrange':  (0.05, 80), # r-range for G(r) calculation/fitting [min, max]
    'myrstep': 0.05,        # Grid spacing for the PDF calculation
    
    'convergence_options': {'disp': True}, # Verbose output from the optimiser
    
    # PDFgetX Processing Defaults
    # Configures underlying variables regulating mathematical scaling if directly processing scattering arrays to intermediate G(r) traces.
    #   - 'mode': Foundational form factor profile designation (typically 'xray' or 'neutron').
    #   - 'dataformat': Designation of the standard input mapping domain. 'QA' designates that the incoming dataset array is already integrated into Q-space.
    #   - 'rpoly': Defines the mathematical degree of the r-polynomial applied dynamically to smoothly correct artificial background deviations at very low r.
    #   - 'qmin': Minimum momentum transfer threshold.
    'pdfgetx_config': {
        'mode':       'xray',
        'dataformat': 'QA', 
        'rpoly':      1.3,  
        'qmin':       0.0
    },
    
    # Special Structure Fallback
    # Pre-evaluated initial model fallback if no previous structural trace exists.
    'special_structure': {
        'file_path': 'CIFs/preoptimisedPDF_ZrV2O7_061_100C_avg_226_245_00000.cif',
        'phase_index_to_update': 0 
    },
    
    # Log and State Persistence
    'log_file': 'refinement_log.txt',
    
    # Dataset Initialisation Methodology
    # If True: initialises with base configuration. If False: sequentially builds upon preceding fits.
    'start_each_dataset_fresh': False,
    
    # Global Optimiser Settings
    #   - 'optimizer': Selects the algorithmic foundation from computational libraries (e.g., 'minimize', 'least_squares', or global approaches like 'basinhopping').
    #   - 'optimizer_method': Denotes the targeted algorithmic formulation, notably 'L-BFGS-B' (limited-memory quasi-Newton solver specifically enforcing bounds).
    #   - 'basinhopping_options': Sub-parameters active merely if a global 'basinhopping' framework structurally defaults. Controls coordinate perturbation ('stepsize'), trial limit ('niter'), and standard energetic descent ('T').
    'optimizer':        'minimize', 
    'optimizer_method': 'L-BFGS-B', 
    'basinhopping_options': {'stepsize': 50, 'niter': 100, 'T': 200.0},
}

# =============================================================================
# 2. REFINEMENT PLAN FOR SEQUENTIAL WORKFLOW
# =============================================================================
# This sequence structurally formalises the chronological refinement strategy progressively applied to the models.
# 
# Parameter Definitions:
#   - 'space_group': The explicitly designated crystallographic symmetry constraining the geometric fit during this step.
#   - 'force_cubic_lattice': Boolean directive. If True, automatically enforces lattice equivalence (forces strict cubic scaling across the a, b, c axes) uniformly despite progressive symmetry dropping.
#   - 'constraints': Geometrical weight penalty array explicitly defined as a tuple (Boolean active variable, rigorous mathematical tolerance constraint parameter limit).
#                    If 'adaptive': True, constraints are algorithmically eased during progressive analytical stages mapped proportionally to final fit residuals.
#   - 'fitting_range': Systematically outlines discrete [r_min, r_max] radial distribution bounds rigorously targeting the quantitative calculation envelope.
#   - 'fitting_order': Sub-sequence dynamically controlling the specific temporal activation of variables. Mapped chronologically per sequential phase.
#                      E.g., 'lat' (lattice coordinate scalar), 'scale', 'psize' (nanoparticle spherical morphology), 'delta2', 'adp' (thermal), 'xyz' (fractional coordinates), culminating mathematically into 'all'.

refinement_plan = {
    0: {
        # Step 0: Establishes a baseline fit using the initial high-symmetry space group (Pa-3).
        'description': 'Initial baseline fit with Pa-3 symmetry and standard constraints',
        'space_group': ['Pa-3'],
        'force_cubic_lattice': True,  
        'constraints': {'constrain_bonds': (True, 0.001), 'constrain_angles': (True, 0.001), 'constrain_dihedrals': (False, 0.001), 'adaptive': True},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    1: {
        # Step 1: Partially reduces symmetry to P23, relaxing specific structural rigidities.
        # This breaks inversion symmetry, allowing distinct atomic subgroups to shift and form
        # local structural distortions not permitted under the strict Pa-3 geometric space.
        'description': 'Partial symmetry reduction to P23 to probe intermediate distortions',
        'space_group': ['P23'],
        'force_cubic_lattice': True,  # Explicitly keep the lattice cubic
        'constraints': {'constrain_bonds': (True, 0.001), 'constrain_angles': (True, 0.001), 'constrain_dihedrals': (False, 0.001), 'adaptive': True},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    2: {
        # Step 2: Removes all crystallographic symmetry rules by dropping down to P1.
        # This permits every single atom in the supercell to move fully independently, providing 
        # the absolute highest degree of freedom to capture localised short-range order dynamics.
        'description': 'Full symmetry relaxation to P1 for unconstrained independent atomic refinement',
        'space_group': ['P1'],
        'force_cubic_lattice': True,  # Explicitly keep the lattice cubic
        'constraints': {'constrain_bonds': (True, 0.001), 'constrain_angles': (True, 0.001), 'constrain_dihedrals': (False, 0.001), 'adaptive': True},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    }
}

# Integrate the defined plan into the main configuration dictionary
project_config['refinement_plan'] = refinement_plan

# =============================================================================
# 3. LIBRARY IMPORTS AND ENVIRONMENT SETUP
# =============================================================================

import matplotlib
import psutil
import multiprocessing
import numpy as np
from multiprocessing import Pool
import sys

# Core custom classes for the PDF structural refinement framework
from pdf_refinement_core import (
    RefinementConfig,
    StructureAnalyzer,
    ResultsManager,
    PDFManager,
    RefinementHelper,
    PDFRefinement,
    PDFWorkflowManager
)

# Set default figure dimensions for exported plots
matplotlib.rc('figure', figsize=(5, 3.75))

# =============================================================================
# 4. SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':

    # --- 1. CONFIGURATION LOADING ---
    # Initialise the central configuration object to validate all input parameters.
    try:
        config = RefinementConfig(project_config)
        print("Configuration loaded successfully.")
    except KeyError as e:
        print(f"Error initialising configuration: {e}")
        sys.exit()

    # --- 2. MULTIPROCESSING ENVIRONMENT ---
    # Automatically determine the number of available CPU cores based on system load.
    syst_cores = multiprocessing.cpu_count()
    cpu_percent = psutil.cpu_percent()
    # Estimate available bandwidth to avoid oversaturating the workstation.
    avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
    ncpu = int(np.max([1, avail_cores]))
    print(f"Assigning {ncpu} cores for parallel processing.")
    
    # Initialise the process pool for parallel PDF calculations.
    pool = Pool(processes=ncpu)

    # --- 3. WORKFLOW COMPONENT INITIALISATION ---
    # Instantiate the components that handle analysis, results management, 
    # and PDF generation respectively.
    analyzer = StructureAnalyzer(config.detailed_composition)
    results_manager = ResultsManager(config, analyzer)
    pdf_manager = PDFManager(config, ncpu, pool)
    helper = RefinementHelper()

    # --- 4. ORCHESTRATOR SETUP ---
    # The 'PDFWorkflowManager' coordinates the communication between all components.
    workflow_orchestrator = PDFWorkflowManager(
        config, pdf_manager, results_manager, helper, analyzer, ncpu, pool
    )
    
    # --- 5. EXECUTION: SEQUENTIAL REFINEMENT ---
    # Initiates the primary workflow running sequential fitting stages defined in the plan.
    print("\n--- Starting Sequential Workflow ---")
    workflow_orchestrator.run_sequential_workflow()

    print("\nScript execution finished.")
# =============================================================================
#                               END OF SCRIPT
# =============================================================================
