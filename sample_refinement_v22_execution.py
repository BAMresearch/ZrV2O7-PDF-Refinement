"""
================================================================================
 Automated Structural Refinement via Pair Distribution Function (PDF) Analysis
================================================================================

@author:    Tomasz Stawski
@contact:   tomasz.stawski@gmail.com | tomasz.stawski@bam.de
@version:   1.0
@date:      2025-09-09

DESCRIPTION:
This script serves as the main execution environment for performing automated,
multi-stage structural refinements of crystalline materials against X-ray
scattering data. It leverages an object-oriented framework defined in the
accompanying 'sample_refinement_v21_classes.py' module.

The workflow is organized into several distinct stages:

1.  **Configuration**: All experimental parameters, file paths, and model
    details are defined in a centralized dictionary (`project_data`).

2.  **Initialization**: The script instantiates the necessary controller and
    manager classes from the imported module, configuring the environment for
    parallel processing to optimize computational performance.

3.  **Data Processing**: The raw experimental scattering data is processed to
    generate the Pair Distribution Function (PDF), G(r), which is the basis
    for the refinement.

4.  **Model Construction**: A theoretical PDF is calculated from an initial
    structural model provided in a Crystallographic Information File (.cif).
    This forms the initial `FitRecipe` for the refinement.

5.  **Sequential Refinement**: The core of the script executes a series of
    refinement steps. The theoretical model is iteratively fitted to the
    experimental PDF by adjusting structural parameters (e.g., lattice
    constants, atomic positions, atomic displacement parameters). The workflow
    is designed to systematically explore different structural symmetries and
    apply chemically motivated rigid-body constraints.

6.  **Simulation**: As a final validation step, the script can run a simulation
    using a previously optimized structure to calculate a theoretical PDF and
    compare it against experimental data.
"""



# =============================================================================
# 1. UNIFIED REFINEMENT CONFIGURATION
# =============================================================================
# This dictionary contains all parameters for the refinement workflow.
# It supports single or multiple datasets via the 'dataset_list'.

project_config = {
    'project_name': 'ZirconiumVanadate_RefinementTest14092025/',
    'xrd_directory': 'data/',
    'cif_directory': 'CIFs/',
    'fit_directory': 'fits/',
    
    # Use 'dataset_list' for one or more files.
    'dataset_list': [
        'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat',
        'PDF_ZrV2O7_061_60C_avg_66_85_00000.dat',
        'PDF_ZrV2O7_061_427C_avg_446_465_00000.dat'
    ],
    
    'ciffile': {'98-005-9396_ZrV2O7.cif': ['Pa-3', True, (1, 1, 1)]},
    'composition': 'O7 V2 Zr1',
    'detailed_composition': {
        'Zr': {'symbol': 'Zr', 'Uiso': 0.0065, 'polyhedron_center': True, 'polyhedron_vertex': False, 'cutoff': (1.8, 2.2)},
        'V':  {'symbol': 'V', 'Uiso': 0.0100, 'polyhedron_center': True, 'polyhedron_vertex': False, 'cutoff': (1.5, 2.4)},
        'O':  {'symbol': 'O', 'Uiso': 0.0250, 'polyhedron_center': False, 'polyhedron_vertex': True},
    },
    'qdamp': 2.70577268e-02,
    'qbroad': 2.40376789e-06,
    'qmax': 22.0,
    'anisotropic': False,
    'unified_Uiso': True,
    'sgoffset': [0.0, 0.0, 0.0],
    'myrange': (0.0, 80),
    'myrstep': 0.05,
    'convergence_options': {'disp': True 
                            , 'ftol': 1e-4 #Stops when the change in Rw is less than 'ftol'
                            },
    'pdfgetx_config': {
        'mode': 'xray',
        'dataformat': 'QA',
        'rpoly': 1.3,
        'qmin': 0.0
    },
    
    # This optional section specifies a pre-refined structure.
    # It will ONLY be used for the first dataset if no checkpoint exists.
    'special_structure': {
        'file_path': 'fits/ZirconiumVanadate25Cperiodic/18032025_071408/Phase0_6.cif',
        'phase_index_to_update': 0 
    },
    
    # Parameters for the sequential/resumable workflow
    'log_file': 'refinement_log.txt',
}

# =============================================================================
# 2. SIMULATION-SPECIFIC PARAMETERS
# =============================================================================
# This dictionary contains parameters exclusively for the final simulation and
# validation workflow.
simulation_data = {
    'cif_directory': 'optimised_PDF_fits_vs_Temp/25C_Phase0_6/',
    'ciffile': {'opt_25C_Phase0_6.cif': ['P1', True, (1, 1, 1)]},
    'powder_data_file': 'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat',
    'output_path': 'resultsSimulations/25C_Phase0_6',
    'optimized_params': {
        'Phase0': {'s': 4.92399836e-01, 'psize': 2.66658626e+02, 'delta2': 2.53696631e+00}
    },
    'default_Uiso': {
        'Zr': 5.79086780e-04,
        'V': 3.21503909e-03,
        'O': 7.21519003e-03
    },
    'fitting_range': [1.5, 27],
    'csv_filename': 'sim_vs_obs.csv'
}

# =============================================================================
# 3. REFINEMENT PLAN FOR SEQUENTIAL WORKFLOW
# =============================================================================
#This dictionary defines the entire multi-step refinement strategy.
refinement_plan = {
    0: {
        'description': 'Initial fit with Pa-3 symmetry and standard constraints',
        'space_group': ['Pa-3'],
        'constraints': {'constrain_bonds': (True, 0.001), 'constrain_angles': (True, 0.001), 'constrain_dihedrals': (False, 0.001), 'adaptive': False},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    1: {
        'description': 'Refinement with tighter constraints (Pa-3 symmetry)',
        'space_group': ['Pa-3'],
        'constraints': {'constrain_bonds': (True, 0.0001), 'constrain_angles': (True, 0.0001), 'constrain_dihedrals': (False, 0.001), 'adaptive': False},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    2: {
        'description': 'Symmetry reduction to P213',
        'space_group': ['P213'],
        'constraints': {'constrain_bonds': (True, 0.001), 'constrain_angles': (True, 0.001), 'constrain_dihedrals': (False, 0.001), 'adaptive': False},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    3: {
        'description': 'Symmetry reduction to P23',
        'space_group': ['P23'],
        'constraints': {'constrain_bonds': (True, 0.001), 'constrain_angles': (True, 0.001), 'constrain_dihedrals': (False, 0.001), 'adaptive': False},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    4: {
        'description': 'Further refinement in P23 with tighter constraints',
        'space_group': ['P23'],
        'constraints': {'constrain_bonds': (True, 0.0001), 'constrain_angles': (True, 0.0001), 'constrain_dihedrals': (False, 0.001), 'adaptive': False},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    5: {
        'description': 'Lowest symmetry (P1)',
        'space_group': ['P1'],
        'constraints': {'constrain_bonds': (True, 0.001), 'constrain_angles': (True, 0.001), 'constrain_dihedrals': (False, 0.001), 'adaptive': False},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    },
    6: {
        'description': 'Final refinement in P1 with tightest constraints',
        'space_group': ['P1'],
        'constraints': {'constrain_bonds': (True, 0.0001), 'constrain_angles': (True, 0.0001), 'constrain_dihedrals': (False, 0.001), 'adaptive': False},
        'fitting_range': [1.5, 27],
        'fitting_order': ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    }
}




# Add the defined plan to the main configuration dictionary
project_config['refinement_plan'] = refinement_plan





# =============================================================================
# 4. LIBRARY IMPORTS AND ENVIRONMENT SETUP
# =============================================================================

import matplotlib
import psutil
import multiprocessing
import numpy as np
from multiprocessing import Pool
import sys

# Core custom classes for the PDF structural refinement
from sample_refinement_v22_classes import (
    RefinementConfig,
    StructureAnalyzer,
    ResultsManager,
    PDFManager,
    RefinementHelper,
    PDFRefinement,
    PDFWorkflowManager
)

matplotlib.rc('figure', figsize=(5, 3.75))

# =============================================================================
# 5. SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':

    # --- 1. Load the Unified Configuration ---
    try:
        config = RefinementConfig(project_config)
        print("Configuration loaded successfully.")
    except KeyError as e:
        print(f"Error initializing configuration: {e}")
        sys.exit()

    # --- 2. Setup the Environment ---
    syst_cores = multiprocessing.cpu_count()
    cpu_percent = psutil.cpu_percent()
    avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
    ncpu = int(np.max([1, avail_cores]))
    pool = Pool(processes=ncpu)

    # --- 3. Instantiate Workflow Components ---
    analyzer = StructureAnalyzer(config.detailed_composition)
    results_manager = ResultsManager(config, analyzer)
    pdf_manager = PDFManager(config, ncpu, pool)
    helper = RefinementHelper()

    # --- 4. Instantiate and Run the Orchestrator ---
    # Corrected to use PDFWorkflowManager
    workflow_orchestrator = PDFWorkflowManager(
        config, pdf_manager, results_manager, helper, analyzer, ncpu, pool
    )
    # Corrected to use the existing method name
    workflow_orchestrator.run_sequential_workflow()

    print("\nScript execution finished.")

    # =============================================================================
    # 8. SIMULATION WORKFLOW (OPTIONAL)
    # =============================================================================
    # This section can be un-commented to execute a final validation simulation
    # using an optimized structural model.

    # workflow.simulate_pdf_workflow(
    #     main_config=config,
    #     sim_config=simulation_data
    # )

    print("\nScript execution finished.")
    # =============================================================================
    #                               END OF SCRIPT
    # =============================================================================
