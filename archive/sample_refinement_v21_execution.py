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
# 1. PRIMARY REFINEMENT CONFIGURATION
# =============================================================================
# This dictionary contains all parameters for the main refinement workflow.
project_data = {
    'project_name': 'ZirconiumVanadate25Cto25C_TESTING_OOP/',
    'xrd_directory': 'data/',
    'cif_directory': 'CIFs/',
    'fit_directory': 'fits/',
    'mypowderdata': 'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat',
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
    'convergence_options': {'disp': True},
    'pdfgetx_config': {
        'mode': 'xray',
        'dataformat': 'QA',
        'rpoly': 1.3,
        'qmin': 0.0
    },
    # This optional section specifies a pre-refined structure to initialize atomic coordinates.
    'special_structure': {
        'file_path': 'fits/ZirconiumVanadate25Cperiodic/18032025_071408/Phase0_6.cif',
        'phase_index_to_update': 0  # 0 corresponds to 'Phase0'
    }
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
# 3. LIBRARY IMPORTS AND ENVIRONMENT SETUP
# =============================================================================

import matplotlib
import psutil
import multiprocessing
import numpy as np
from multiprocessing import Pool

# Core custom classes for the PDF structural refinement
from sample_refinement_v21_classes import (
    RefinementConfig,
    StructureAnalyzer,
    ResultsManager,
    PDFManager,
    RefinementHelper,
    PDFRefinement
)

matplotlib.rc('figure', figsize=(5, 3.75))

# =============================================================================
# 4. SCRIPT EXECUTION
# =============================================================================

if __name__ == '__main__':

    # 4.1. Configuration and Component Initialization
    try:
        config = RefinementConfig(project_data)
        print("Configuration loaded successfully.")
    except KeyError as e:
        print(f"Error initializing configuration: {e}")
        exit()

    # Configure parallel processing based on available system resources.
    syst_cores = multiprocessing.cpu_count()
    cpu_percent = psutil.cpu_percent()
    avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
    ncpu = int(np.max([1, avail_cores]))
    pool = Pool(processes=ncpu)

    # Instantiate all necessary manager and controller classes.
    analyzer = StructureAnalyzer(config.detailed_composition)
    results_manager = ResultsManager(config, analyzer)
    pdf_manager = PDFManager(config, ncpu, pool)
    helper = RefinementHelper()

    # The PDFRefinement class serves as the main controller, aggregating other components.
    workflow = PDFRefinement(config, pdf_manager, results_manager, helper, analyzer, ncpu, pool)

    # 4.2. Initial Data Processing and Model Construction
    print("\nGenerating initial PDF from experimental data...")
    r0, g0, cfg = pdf_manager.generatePDF(
        data_directory=config.xrd_directory,
        data_filename=config.mypowderdata,
        composition=config.composition,
        qmax=config.qmax,
        myrange=config.myrange,
        myrstep=config.myrstep,
        pdfgetx_config=config.pdfgetx_config
    )

    # Build the initial PDFContribution and FitRecipe objects.
    cpdf0 = pdf_manager.build_contribution(r0, g0, cfg, config.ciffile, config.myrange)
    workflow.cpdf = cpdf0
    fit0 = workflow.build_initial_recipe()

    # 4.3. Initialization from a Pre-Refined Structure
    # This step updates the atomic coordinates in the initial model using those
    # from a previously refined, low-symmetry structure. This provides a more
    # accurate starting point for the refinement by breaking initial symmetry.
    workflow.initialize_from_special_structure()

    # =============================================================================
    # 5. SEQUENTIAL REFINEMENT WORKFLOW (FIRST DATASET)
    # =============================================================================
    # The following blocks define a multi-stage refinement strategy. Parameters
    # are refined sequentially, and crystal symmetry is progressively lowered
    # to test for better-fitting, less-symmetric structural models.

    fitting_order = ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    fitting_range = [1.5, 27]

    # --- Step 0: Initial Fit (Pa-3 symmetry) ---
    i = 0
    workflow.modify_recipe_spacegroup(['Pa-3'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 1: Refinement with Tighter Constraints (Pa-3 symmetry) ---
    i = 1
    workflow.modify_recipe_spacegroup(['Pa-3'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 2: Symmetry Reduction (P213) ---
    i = 2
    workflow.modify_recipe_spacegroup(['P213'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 3: Symmetry Reduction (P23) ---
    i = 3
    workflow.modify_recipe_spacegroup(['P23'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 4: Further Refinement (P23 symmetry) ---
    i = 4
    workflow.modify_recipe_spacegroup(['P23'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 5: Lowest Symmetry (P1) ---
    i = 5
    workflow.modify_recipe_spacegroup(['P1'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 6: Final Refinement (P1 symmetry) ---
    i = 6
    workflow.modify_recipe_spacegroup(['P1'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # Finalize and save all results from the first refinement series.
    results_manager.finalize_results(workflow.cpdf, workflow.fit)

    # =============================================================================
    # 6. SEQUENTIAL REFINEMENT WORKFLOW (SECOND DATASET)
    # =============================================================================
    # This section demonstrates how to continue the refinement process using a
    # new dataset while carrying over the results from the previous stage.

    print("\nInitiating second refinement stage with a new dataset...")
    config.mypowderdata = 'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat'
    config.new_output_directory()
    workflow.results_manager = ResultsManager(config, analyzer)

    # Load the new experimental data.
    r0_2, g0_2, cfg_2 = pdf_manager.generatePDF(
        data_directory=config.xrd_directory,
        data_filename=config.mypowderdata,
        composition=config.composition,
        qmax=config.qmax,
        myrange=config.myrange,
        myrstep=config.myrstep,
        pdfgetx_config=config.pdfgetx_config
    )

    # Create a new contribution using the refined structure from the previous
    # stage and the newly loaded data.
    cpdf1 = workflow.rebuild_contribution(
        old_cpdf=workflow.cpdf, r_obs=r0_2, g_obs=g0_2
    )

    # Build a new, default recipe for the second stage.
    workflow.cpdf = cpdf1
    fit1 = workflow.build_initial_recipe()

    # CRITICAL STEP: Update the new recipe (`fit1`) with the refined parameter
    # values from the end of the first stage (`fit0`).
    workflow.update_recipe_from_initial(fit0, fit1, cpdf1, recalculate_bond_vectors=True)

    # --- Refinement steps for the second dataset ---
    fitting_range = [1.5, 27]
    fitting_order = ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']

    # --- Step 0: Initial Fit (new data) ---
    i = 0
    workflow.modify_recipe_spacegroup(['Pa-3'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 1: Refinement (new data) ---
    i = 1
    workflow.modify_recipe_spacegroup(['Pa-3'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 2: Adjust Symmetry (new data) ---
    i = 2
    workflow.modify_recipe_spacegroup(['P213'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=False)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 3: Adjust Symmetry (new data) ---
    i = 3
    workflow.modify_recipe_spacegroup(['P23'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=True)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 4: Further Refinement (new data) ---
    i = 4
    workflow.modify_recipe_spacegroup(['P23'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=True)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 5: Lowest Symmetry (new data) ---
    i = 5
    workflow.modify_recipe_spacegroup(['P1'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=True)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # --- Step 6: Lowest Symmetry (new data) ---
    i = 6
    workflow.modify_recipe_spacegroup(['P1'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=True)
    workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # Finalize and save all results from the second refinement series.
    workflow.results_manager.finalize_results(workflow.cpdf, workflow.fit)

    # =============================================================================
    # 7. SIMULATION WORKFLOW (OPTIONAL)
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
