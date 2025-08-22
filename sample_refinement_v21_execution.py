"""
Created on Thu Nov 23 12:39:22 2023

@author: Tomasz Stawski
tomasz.stawski@gmail.com
tomasz.stawski@bam.de

# =============================================================================
# DESCRIPTION:
This script automates the process of refining a crystal structure against experimental X-ray scattering data. 

Here is a breakdown of its workflow:

Configuration: The script starts by defining all experimental parameters in a project_data dictionary. This includes file paths, chemical composition, and instrumental settings.

Initialization: It then uses a set of custom classes (imported from sample_refinement_v21_classes) to set up the refinement environment. This includes preparing for parallel processing to speed up calculations.

Data Loading: The script processes the raw experimental data to generate a Pair Distribution Function, G(r), which represents the probability of finding atom pairs at a certain distance r.

Model Building: It builds a theoretical PDF model based on an initial crystal structure guess from a .cif file.

Refinement (Optional/Commented Out): The core of the script is a multi-step refinement process where the theoretical model is fitted to the experimental data. The script is set up to:

Refine parameters like lattice constants, atomic positions, and atomic vibration amplitudes.

Systematically lower the crystal symmetry (e.g., from cubic Pa-3 to triclinic P1) to see if a less symmetric model provides a better fit.

Apply rigid-body constraints to maintain realistic chemical bonding (bond lengths and angles) during refinement.

Simulation: Finally, the script runs a simulation workflow. It takes a previously optimized structure and calculates a theoretical PDF, comparing it against the experimental data to validate the model's accuracy.
"""

#=============================================================================
    #                         PROJECT SETUP & INITIALIZATION
    # =============================================================================
    
project_data = {
'project_name': 'ZirconiumVanadate25Cto25C_TESTING/',
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
    'qmin': 0.0},
'special_structure': {
    'file_path': 'fits/ZirconiumVanadate25Cperiodic/18032025_071408/Phase0_6.cif',
    'phase_index_to_update': 0}  # 0 for Phase0, 1 for Phase1, etc.
}


# =============================================================================
# load libraries and settings 
# =============================================================================

# Configure matplotlib for plotting
import matplotlib

# =============================================================================
# Set the default figure size to 5 x 3.75 inches for consistency.
matplotlib.rc('figure', figsize=(5, 3.75))

# =============================================================================
# Enable multi-threaded processing for parallel computations
# Uses multiprocessing and psutil to manage CPU usage efficiently.
import psutil
import multiprocessing
from multiprocessing import Pool, cpu_count

# =============================================================================
# Numerical operations and arrays.
import numpy as np  

# =============================================================================
# Core custom classes for the PDF structural refinement
from sample_refinement_v21_classes import (
    RefinementConfig,
    StructureAnalyzer,
    ResultsManager,
    PDFManager,
    RefinementHelper,
    PDFRefinement
)


# =============================================================================
#                             MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    
    # 1. Load all settings from the configuration class
    try:
        config = RefinementConfig(project_data)
        print("Configuration loaded successfully.")
    except KeyError as e:
        print(f"Error initializing configuration: {e}")
        # Exit or handle the error appropriately
        exit()
        
    # 2. Set up multiprocessing
    syst_cores = multiprocessing.cpu_count()
    cpu_percent = psutil.cpu_percent()
    avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
    ncpu = int(np.max([1, avail_cores]))
    pool = Pool(processes=ncpu)

    # 3. Instantiate all the necessary manager and workflow classes
    analyzer = StructureAnalyzer(config.detailed_composition)
    results_manager = ResultsManager(config, analyzer)
    pdf_manager = PDFManager(config, ncpu, pool)
    helper = RefinementHelper()
    
    # The main workflow controller holds instances of the other classes
    workflow = PDFRefinement(config, pdf_manager, results_manager, helper, analyzer, ncpu, pool)

    # 4. Generate the initial PDF from experimental data
    r0, g0, cfg = pdf_manager.generatePDF(
        data_directory=config.xrd_directory,
        data_filename=config.mypowderdata,
        composition=config.composition,
        qmax=config.qmax,
        myrange=config.myrange,
        myrstep=config.myrstep,
        pdfgetx_config=config.pdfgetx_config
    )

    # 5. Build the PDFContribution and the initial FitRecipe
    cpdf = pdf_manager.build_contribution(r0, g0, cfg, config.ciffile, config.myrange)
    workflow.cpdf = cpdf
    fit0 = workflow.build_initial_recipe()
    #--------------------------------------------------------------------
    #                   SPECIAL STRUCTURE MODIFICATION
    #----------------------------------------------------------------------
    # Here we load a “special” refined structure from a previous PDF refinement
    # (e.g. the result of refining Phase0 at 25 °C in space group P1, replicate (1×1×1)).
    # This refined CIF contains updated atomic positions that broke higher symmetry
    # constraints in order to capture subtle distortions at this temperature.
    
    #workflow.initialize_from_special_structure()

    # =============================================================================
    #                               FITTING STEPS (FIRST DATASET)
    # =============================================================================

    fitting_order = ['lat', 'scale', 'psize', 'delta2', 'adp', 'xyz', 'all']
    fitting_range = [1.5, 27]
    
    # ========================== Step 0: Initial Fit =============================
    i = 0
    workflow.modify_recipe_spacegroup(['Pa-3'])
    workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001))
    #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 1: Refinement ==============================
    # i = 1
    # workflow.modify_recipe_spacegroup(['Pa-3'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001))
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 2: Adjust Symmetry =========================
    # i = 2
    # workflow.modify_recipe_spacegroup(['P213'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001))
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 3: Adjust Symmetry =========================
    # i = 3
    # workflow.modify_recipe_spacegroup(['P23'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001))
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 4: Further Refinement ======================
    # i = 4
    # workflow.modify_recipe_spacegroup(['P23'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001))
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 5: Lowest Symmetry =========================
    # i = 5
    # workflow.modify_recipe_spacegroup(['P1'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001))
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 6: Lowest Symmetry =========================
    # i = 6
    # workflow.modify_recipe_spacegroup(['P1'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001))
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')
    
    # # =============================================================================
    # #                             FINALIZE RESULTS
    # # =============================================================================
    results_manager.finalize_results(workflow.cpdf, workflow.fit)

    # # =============================================================================
    # #                   CONTINUE FITTING WITH DIFFERENT DATA
    # # =============================================================================
    # =============================== New (or the same) XRD Data ====================================
    config.mypowderdata = 'PDF_ZrV2O7_061_25C_avg_46_65_00000.dat'
    
    config.new_output_directory()
    # Update the results manager to use the new output path
    workflow.results_manager = ResultsManager(config, analyzer)
    
    # Load the new data
    # CORRECTED THIS FUNCTION CALL
    r0_2, g0_2, cfg_2 = pdf_manager.generatePDF(
        data_directory=config.xrd_directory,
        data_filename=config.mypowderdata,
        composition=config.composition,
        qmax=config.qmax,
        myrange=config.myrange,
        myrstep=config.myrstep,
        pdfgetx_config=config.pdfgetx_config
    )
    
    # Build a fresh cpdf2 that reuses the exact same Structures from `cpdf`
    cpdf2 = workflow.rebuild_contribution(
        old_cpdf=workflow.cpdf, r_obs=r0_2, g_obs=g0_2
    )
    
    # Re-initialize a brand-new FitRecipe from cpdf2:
    fit1 = workflow.rebuild_recipe_from_initial(
        fit_old=workflow.fit, cpdf_new=cpdf2, spaceGroups=['Pa-3']
    )

    # # ========================== FITTING STEPS (SECOND DATASET) =================
    
    # fitting_range = [27, 60] # New fitting range
    
    # # ========================== Step 0: Initial Fit (new data) ==================
    # i = 0
    # workflow.modify_recipe_spacegroup(['Pa-3'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=True)
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 1: Refinement (new data) ====================
    # i = 1
    # workflow.modify_recipe_spacegroup(['Pa-3'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=True)
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 2: Adjust Symmetry (new data) ===============
    # i = 2
    # workflow.modify_recipe_spacegroup(['P213'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=True)
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')
    
    # # ========================== Step 3: Adjust Symmetry (new data) ===============
    # i = 3
    # workflow.modify_recipe_spacegroup(['P23'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=True)
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 4: Further Refinement (new data) ===========
    # i = 4
    # workflow.modify_recipe_spacegroup(['P23'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=True)
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 5: Lowest Symmetry (new data) ==============
    # i = 5
    # workflow.modify_recipe_spacegroup(['P1'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.001), constrain_angles=(True, 0.001), constrain_dihedrals=(False, 0.001), adaptive=True)
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')

    # # ========================== Step 6: Lowest Symmetry (new data) ==============
    # i = 6
    # workflow.modify_recipe_spacegroup(['P1'])
    # workflow.apply_rigid_body_constraints(constrain_bonds=(True, 0.0001), constrain_angles=(True, 0.0001), constrain_dihedrals=(False, 0.001), adaptive=True)
    # #workflow.run_refinement_step(i, fitting_range, config.myrstep, fitting_order, 'resv')
    
    # # =============================================================================
    # #                             FINALIZE RESULTS (new data)
    # # =============================================================================
    # workflow.results_manager.finalize_results(workflow.cpdf, workflow.fit)
    
    #=============================================================================
    # Run simulation workflow with specific parameters
    #=============================================================================
    
# =============================================================================
#                 SIMULATION WORKFLOW CONFIGURATION
# =============================================================================
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
#                       RUN SIMULATION WORKFLOW
# =============================================================================

# Execute the simulation with the organized configuration
# workflow.simulate_pdf_workflow(
#     # Pass general config for shared parameters like composition, q-values, etc.
#     main_config=config,
#     # Pass the dedicated dictionary for simulation-specific settings
#     sim_config=simulation_data
# )



    
# =============================================================================
    # End of script
    # =============================================================================
