"""
================================================================================
  Instrumental Calibration via LaB6 Structural Refinement
================================================================================

Author: Tomasz Stawski (tomasz.stawski@bam.de, tomasz.stawski@gmail.com)
Testing, Implementation and Analyses: Aiste Miliute (aiste.miliute@bam.de)

Version: 1.3.0
License: MIT License

DESCRIPTION:
This script calibrates internal instrumental parameters (qdamp and qbroad) by 
refining a known structural standard (LaB6) against empirical 
X-ray scattering data. 
"""

import sys
import os
import multiprocessing
import psutil
import numpy as np
import matplotlib
from multiprocessing import Pool

from pdf_refinement_core import (
    RefinementConfig,
    StructureAnalyzer,
    ResultsManager,
    PDFManager,
    RefinementHelper,
    PDFWorkflowManager
)

matplotlib.rc('figure', figsize=(5, 3.75))

# =============================================================================
# 1. CORE CALIBRATION CONFIGURATION
# =============================================================================
# This dictionary ('project_config') holds the global settings and structural 
# definitions required by the underlying 'pdf_refinement_core' classes.
project_config = {
    # Project and Directory Management
    'project_name': 'LaB6_Calibration/',         # Root name for output folders
    'xrd_directory': 'LaB6 standard/data/',       # Location of scattering data
    'data_dir_gr': 'LaB6 standard/dataXRDandGofRs/',  # Location of precalculated .gr files (Fallback)
    'cif_directory': 'LaB6 standard/CIFs/',       # Location of structural models (.cif)
    'fit_directory': 'LaB6 standard/fits/',       # Default output for refinement results
    
    # Dataset Selection
    # Specify discrete calibration datasets for sequential processing.
    'dataset_list': [
        'LaB6_SDD300_Calib_DAWN_00000.dat'
    ],
    
    # Structural and Chemical Definitions
    # 'ciffile' dictionary sets the structural starting metric.
    # [Space Group, Periodic Boundary Expansion, (Supercell multiples)]
    'ciffile': {'LaB6_NISTSRM_660b.cif': ['Pm3m', True, (1, 1, 1)]}, 
    'composition': 'B6 La', # Overall empirical stoichiometry
    
    # Detailed Atom-Specific Settings
    # Dictates baseline variables used independently for geometry and forensics.
    'detailed_composition': {
        'La': {'symbol': 'La', 'Uiso': 0.05, 'polyhedron_center': False,  'polyhedron_vertex': False, 'cutoff': (0.0, 0.0)},
        'B':  {'symbol': 'B',  'Uiso': 0.05, 'polyhedron_center': False,  'polyhedron_vertex': False, 'cutoff': (0.0, 0.0)},
    },
    
    # Instrument and Resolution Parameters
    # Estimated initial values for baseline factors prior to active refinement.
    'qdamp':  2.70e-02, 
    'qbroad': 2.40e-06, 
    'qmax':   22.0,           
    
    # Active Refinement Controls
    # Conditionally directs if instrumental parameters are actively targeted.
    'refine_qdamp':  True, 
    'refine_qbroad': True, 
    
    # Atomic Displacement Parameter (ADP) Handling Strategy
    'adp_mode': 'anisotropic',
    'sgoffset': [0.0, 0.0, 0.0], # Space group origin offset, if applicable
    
    # Calculation Range and Step Size (Å)
    'myrange':  (0.05, 100.0), # r-range for G(r) calculation/fitting [min, max]
    'myrstep': 0.01,           # Grid spacing for the PDF calculation
    'use_shape_envelope': False, # Completely disables mathematical nanoparticle size attenuation
    
    'convergence_options': {'disp': True}, # Verbose output from the optimiser
    
    # PDFgetX Processing Defaults
    'pdfgetx_config': {
        'mode':       'xray',
        'dataformat': 'QA', 
        'rpoly':      1.3,  
        'qmin':       0.0
    },
    
    # Log and State Persistence
    'log_file': 'calibration_log.txt',
    'start_each_dataset_fresh': True, # Forces recalibration independently per dataset
    
    # Global Optimiser Settings
    'optimizer':        'minimize', 
    'optimizer_method': 'L-BFGS-B', 
    'basinhopping_options': {'stepsize': 50, 'niter': 100, 'T': 200.0},
}

# =============================================================================
# 2. CALIBRATION REFINEMENT PLAN
# =============================================================================
# This sequence formally structures the calibration strategy applied progressively.
# It isolates parameters iteratively rather than refining them simultaneously, 
# ensuring mathematical stability when determining qdamp and qbroad.
refinement_plan = {
    0: {
        'description': 'Full-range refinement targeting qdamp',
        'space_group': ['Pm3m'],
        'force_cubic_lattice': False,  
        'constraints': {'constrain_bonds': (False, 0.0), 'constrain_angles': (False, 0.0), 'constrain_dihedrals': (False, 0.0), 'adaptive': False},
        'fitting_range': [1.5, 100.0],
        'fitting_order': ['lat', 'scale', 'delta2', 'adp', 'xyz', 'qdamp']
    },
    1: {
        'description': 'Full-range refinement isolating qbroad variance',
        'space_group': ['Pm3m'],
        'force_cubic_lattice': False, 
        'constraints': {'constrain_bonds': (False, 0.0), 'constrain_angles': (False, 0.0), 'constrain_dihedrals': (False, 0.0), 'adaptive': False},
        'fitting_range': [1.5, 100.0],
        'fitting_order': ['lat', 'scale', 'delta2', 'adp', 'xyz', 'qbroad']
    }
}

project_config['refinement_plan'] = refinement_plan

# =============================================================================
# 3. SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':

    # --- 1. CONFIGURATION LOADING ---
    try:
        config = RefinementConfig(project_config)
        print("Calibration configuration loaded successfully.")
    except KeyError as e:
        print(f"Error initialising configuration: {e}")
        sys.exit()

    # --- 2. MULTIPROCESSING ENVIRONMENT ---
    syst_cores = multiprocessing.cpu_count()
    cpu_percent = psutil.cpu_percent()
    avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
    ncpu = int(np.max([1, avail_cores]))
    print(f"Assigning {ncpu} cores for parallel processing.")
    
    pool = Pool(processes=ncpu)

    # --- 3. WORKFLOW COMPONENT INITIALISATION ---
    analyzer = StructureAnalyzer(config.detailed_composition)
    results_manager = ResultsManager(config, analyzer)
    pdf_manager = PDFManager(config, ncpu, pool)
    helper = RefinementHelper()

    # --- 4. ORCHESTRATOR SETUP ---
    workflow_orchestrator = PDFWorkflowManager(
        config, pdf_manager, results_manager, helper, analyzer, ncpu, pool
    )
    
    # --- 5. EXECUTION: SEQUENTIAL CALIBRATION ---
    print("\n--- Starting Sequential Calibration Workflow ---")
    workflow_orchestrator.run_sequential_workflow()

    print("\nCalibration execution finished.")
# =============================================================================
#                               END OF SCRIPT
# =============================================================================
