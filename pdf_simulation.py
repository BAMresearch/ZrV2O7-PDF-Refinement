"""
================================================================================
  Structural Simulation and Post-Fit Forensics Analysis
================================================================================

Author: Tomasz Stawski (tomasz.stawski@bam.de, tomasz.stawski@gmail.com)
Testing, Implementation and Analyses: Aiste Miliute (aiste.miliute@bam.de)
Support and Input: Joana Bustamante (joana.bustamante@bam.de)
Version: 1.3.0
License: MIT License

DESCRIPTION:
This script performs the final simulation and analysis phase from a completed 
Pair Distribution Function (PDF) structural refinement. It separates 
post-refinement validation from the main fitting workflow.

The script provides the following functionalities:

1.  **Simulation**: Validates an optimised structural model by generating a 
    theoretical PDF using final refined parameters (e.g., scale, particle 
    size, delta2) and comparing it against the experimental data.

2.  **Structural Forensics**: Analyses a given CIF model (such as a 
    checkpoint or a final fitted structure) to identify bond lengths and 
    structural distortions, highlighting any physically unrealistic outliers 
    (e.g., V-O bonds > 1.8 Å).
"""

# =============================================================================
# 1. CORE PROJECT CONFIGURATION
# =============================================================================
# This dictionary ('project_config') holds the global settings and structural 
# definitions required by the underlying 'pdf_refinement_core' classes.
# While this script focuses on simulation, it still requires some f these basic 
# parameters to reconstruct the refinement environment correctly.

project_config = {
    # Project and Directory Management
    'project_name':  'ZirconiumVanadate_RefinementTest/', # Root name for output folders
    'xrd_directory': 'dataXRD/',                          # Location of scattering data
    'data_dir_gr':   'dataXRDandGofRs/',                    # Location of precalculated .gr files (Fallback)
    'cif_directory': 'CIFs/',                             # Location of structural models (.cif)
    'fit_directory': 'fits/',                             # Default output for refinement results
    
    # Dataset Selection
    # For standalone simulation, one typically leaves this empty as 'simulation_data' 
    # defines the specific target file.
    'dataset_list': [],
    
    # Structural and Chemical Definitions
    # Defines the starting model and its symmetry (Space Group, periodicity, super-cell)
    'ciffile': {'98-005-9396_ZrV2O7.cif': ['Pa-3', True, (1, 1, 1)]},
    'composition': 'O7 V2 Zr1', # Overall stoichiometry
    
    # Detailed Atom-Specific Settings
    # Defines roles (polyhedron centres vs vertices), initial thermal displacements (Uiso),
    # and geometric cut-offs for bond/angle analysis.
    'detailed_composition': {
        'Zr': {'symbol': 'Zr', 'Uiso': 0.0065, 'polyhedron_center': True,  'polyhedron_vertex': False, 'cutoff': (1.8, 2.2)},
        'V':  {'symbol': 'V',  'Uiso': 0.0100, 'polyhedron_center': True,  'polyhedron_vertex': False, 'cutoff': (1.5, 2.4)},
        'O':  {'symbol': 'O',  'Uiso': 0.0250, 'polyhedron_center': False, 'polyhedron_vertex': True},
    },
    
    # Instrument and Resolution Parameters
    'qdamp':  2.70577268e-02, # PDF resolution factor from thermal/instrumental effects
    'qbroad': 2.40376789e-06, # PDF broadening factor (peak width at high Q)
    'qmax':   22.0,           # Maximum momentum transfer for PDF generation (Å⁻¹)
    
    # Refinement Controls
    'refine_qdamp':  False, # Do not adjust qdamp during analysis
    'refine_qbroad': False, # Do not adjust qbroad during analysis
    
    # Atomic Displacement Parameter (ADP) Handling
    # 'fixed_shape' preserves the anisotropy from the input CIF while scaling the magnitude.
    'adp_mode': 'fixed_shape',
    'sgoffset': [0.0, 0.0, 0.0], # Space group origin offset, if applicable
    
    # Calculation Range and Step Size (Å)
    'myrange':  (0.05, 27), # r-range for G(r) calculation/fitting [min, max]
    'myrstep': 0.05,        # Grid spacing for the PDF calculation
    
    'convergence_options': {'disp': True}, # Verbose output from the optimiser
    
    # PDFgetX Processing Defaults
    # Controls how raw scattering data is converted to G(r)
    'pdfgetx_config': {
        'mode':       'xray', 
        'dataformat': 'QA',   # Q-domain input
        'rpoly':      1.3,    # Background subtraction polynomial degree
        'qmin':       0.0
    },
    
    # Log and State Persistence
    'log_file': 'refinement_log.txt',
    'start_each_dataset_fresh': False, # Continue sequence from last successful result
    
    # Global Optimiser Settings
    'optimizer':        'minimize', # Core optimisation engine
    'optimizer_method': 'L-BFGS-B',  # Specifically, the box-constrained quasi-Newton method
}

# =============================================================================
# 2. SIMULATION-SPECIFIC PARAMETERS
# =============================================================================
# This dictionary ('simulation_data') defines the parameters for a one-off 
# simulation run where we calculate a theoretical G(r) from an existing CIF 
# and compare it against experimental data.

simulation_data = {
    'cif_directory':    'CIFs/',  # Folder containing the model for simulation
    'ciffile': {
        'preoptimisedPDF_ZrV2O7_061_100C_avg_226_245_00000.cif': ['P1', True, (1, 1, 1)]
    }, 
    'powder_data_file': 'PDF_ZrV2O7_061_100C_avg_226_245_00000.dat', # Reference experimental data
    'output_path':      'ExampleOfSimulationsAndAnalyses/resultsSimulations/100C_Phase0_test',
    
    # Refined structural parameters to apply to the model
    'optimized_params': {
        'Phase0': {
            's':      4.92890476e-01, # Scale factor
            'psize':  3.96007725e+02, # Particle size (spherical shape factor)
            'delta2': 2.56211060e+00  # Correlated atomic vibration factor
        }
    },
    
    # Initial/Default isotropic thermal displacement values (Å²)
    'default_Uiso': {
        'Zr': 6.95944235e-06,
        'V':  1.13130649e-03,
        'O':  6.87032735e-03
    },
    'fitting_range': [0.05, 27],      # Range for the residual calculation (Rw)
    'csv_filename':  'sim_vs_obs.csv'  # Output for simulated and observed curves
}

# =============================================================================
# 3. REFINEMENT PLAN STUB
# =============================================================================
# This script does not perform iterative fitting, so the refinement plan 
# is explicitly left empty.
project_config['refinement_plan'] = {}

# =============================================================================
# 4. LIBRARY IMPORTS AND ENVIRONMENT SETUP
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
# 5. SCRIPT EXECUTION
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
    # Estimate available bandwidth to avoid oversaturating the workstation
    avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
    ncpu = int(np.max([1, avail_cores]))
    print(f"Assigning {ncpu} cores for parallel processing.")
    
    # Initialise the process pool for parallel heavy lifting (PDF calculations)
    pool = Pool(processes=ncpu)

    # --- 3. WORKFLOW COMPONENT INITIALISATION ---
    # Instantiate the components that handle analysis, results management, 
    # and PDF generation respectively.
    analyzer = StructureAnalyzer(config.detailed_composition) # Geometrical analysis
    results_manager = ResultsManager(config, analyzer)        # Output/File management
    pdf_manager = PDFManager(config, ncpu, pool)              # Core PDF calculations
    helper = RefinementHelper()                                # Utility functions

    # --- 4. ORCHESTRATOR SETUP ---
    # The 'PDFWorkflowManager' coordinates the communication between all components.
    workflow_orchestrator = PDFWorkflowManager(
        config, pdf_manager, results_manager, helper, analyzer, ncpu, pool
    )
    
    # --- 5. EXECUTION: SIMULATION WORKFLOW ---
    # Calculates a theoretical PDF from the provided parameters and generates 
    # comparison plots against experimental data.
    print("\n--- Starting Simulation ---")
    workflow_orchestrator.simulate_pdf_workflow(
         main_config=config,
         sim_config=simulation_data
     )

    # --- 6. EXECUTION: STRUCTURAL FORENSICS ---
    # Performs an in-depth geometrical audit of a specific CIF file 
    # (checkpoint or final result). This cross-references bond lengths 
    # against outliers and chemical reasonability.
    
    target_cif = "CIFs/preoptimisedPDF_ZrV2O7_061_100C_avg_226_245_00000.cif"
    
    # Part A: Quantitative Text Report
    # Generates summaries of bond lengths by connectivity type and outlier counts.
    workflow_orchestrator.run_structural_forensics(
        cif_path=target_cif,
        target_bond='V-O', 
        outlier_threshold=1.8
    )

    # Part B: Statistical Visualisations
    # Generates histograms, coordination distribution plots, and bond forensics graphs.
    print(f"\n[INFO] Saving forensics visualisations...")
    workflow_orchestrator.visualize_structural_forensics(
        cif_path=target_cif,
        target_bond='V-O',
        outlier_threshold=1.8,
        output_dir="ExampleOfSimulationsAndAnalyses/forensics_results/Dataset_100C" 
    )

    print("\nScript execution finished.")
# =============================================================================
#                               END OF SCRIPT
# =============================================================================
