#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Core Classes for Pair Distribution Function (PDF) Structural Refinement
================================================================================

@author:    Tomasz Stawski
@contact:   tomasz.stawski@gmail.com | tomasz.stawski@bam.de
@version:   1.0
@date:      2025-09-09
@status:    Production

DESCRIPTION:
This module defines a set of classes for performing PDF structural refinements.
The classes are designed to be instantiated and controlled by an external
execution script.

CLASSES:
  - RefinementConfig: Stores and validates all project configuration parameters.
  - StructureAnalyzer: Performs geometric calculations on crystal structures.
  - ResultsManager: Handles the generation and saving of output files and plots.
  - PDFManager: Manages the generation of PDF data and `PDFContribution` objects.
  - RefinementHelper: Contains static helper methods for the refinement workflow.
  - PDFRefinement: The main controller class that integrates all other components
    and executes the refinement workflow.
"""

# =============================================================================
#                          IMPORTS FOR CLASSES
# =============================================================================
# =============================================================================
# Configure matplotlib for plotting
import matplotlib
matplotlib.use('Agg')
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
import dill         # For advanced serializing and deserializing Python objects.
import logging      # For progress logging
import json         # For working with json files
from datetime import datetime  # for time related operations
from itertools import combinations, combinations_with_replacement  # Combinatorial functions

# Enable multi-threaded processing for parallel computations
import threading

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
# Additional utilities for progress tracking and distance calculations
# tqdm: For progress bars
# scipy.spatial.distance: For calculating pairwise distances
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
#                              CLASS DEFINITIONS
# =============================================================================

class RefinementConfig:
    """
    Stores and validates all configuration parameters for a PDF refinement.

    This class is initialized from a dictionary. It verifies that all required
    parameters are present and dynamically creates timestamped output directories
    for storing results.
    """
    def __init__(self, params: dict):
        """
        Initializes the configuration from a dictionary of parameters.
        """
        # Define the keys required for the unified workflow.
        required_keys = [
            'project_name', 'xrd_directory', 'cif_directory', 'fit_directory',
            'dataset_list', # Replaces 'mypowderdata'
            'ciffile', 'composition', 'detailed_composition',
            'qdamp', 'qbroad', 'qmax', 'anisotropic', 'unified_Uiso',
            'sgoffset', 'myrange', 'myrstep', 'convergence_options',
            'pdfgetx_config', 'log_file', 'refinement_plan'
        ]

        # Validate that all required keys are present.
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            raise KeyError(
                f"The following required configuration parameter(s) are missing: {', '.join(missing_keys)}"
            )

        # Dynamically assign all provided parameters as attributes.
        for key, value in params.items():
            setattr(self, key, value)

        # # =============================== Dynamic Output Setup ================================
        # self.time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        # self.output_results_path = os.path.join(
        #     self.fit_directory, self.project_name, self.time_stamp
        # )
        # if not os.path.isdir(self.output_results_path):
        #     os.makedirs(self.output_results_path)
            
            
    def new_output_directory(self, subdir_name=None):
        """
        Creates a new, unique, timestamped directory for a subsequent
        refinement run within the same project.
    
        Args:
            subdir_name (str, optional): A specific name for the subdirectory.
                If None, a timestamp is used.
        """
        if subdir_name:
            self.output_results_path = os.path.join(
                self.fit_directory, self.project_name, subdir_name
            )
        else:
            self.time_stamp = datetime.now().strftime("%d%m%Y%H%M%S")
            self.output_results_path = os.path.join(
                self.fit_directory, self.project_name, self.time_stamp
            )
    
        if not os.path.isdir(self.output_results_path):
            os.makedirs(self.output_results_path)
        print(f"\nSet new output directory: {self.output_results_path}")



class StructureAnalyzer:
    """
    Performs geometric and crystallographic calculations for a structure.

    This class computes structural properties such as bond lengths, angles,
    and dihedrals. It also generates mathematical string expressions used for
    applying rigid-body constraints during a refinement.
    """

    def __init__(self, detailed_composition):
        """
        Initializes the StructureAnalyzer.

        Parameters:
        - detailed_composition: A dictionary containing element-specific information
          like Uiso guesses, roles (center/vertex), and bond cutoffs.
        """
        self.detailed_composition = detailed_composition

    def _lattice_vectors(self, a, b, c, alpha, beta, gamma):
        """
        Calculate the lattice transformation matrix from lattice parameters.
        (Internal helper method)

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

    def calculate_angle(self, v1, v2):
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

    def calculate_dihedral(self, v1, v2, v3):
        """
        Calculate the dihedral angle in degrees between planes defined by v1, v2, and v3.

        Parameters:
        - v1: Vector from atom A1 to A2.
        - v2: Vector from atom A2 to A3.
        - v3: Vector from atom A3 to A4.

        Returns:
        - dihedral_angle: Float, the dihedral angle in degrees. Returns None if vectors are invalid.
        """
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
        x = np.clip(x, -1.0, 1.0)
        dihedral_angle = np.degrees(np.arctan2(y, x))

        return dihedral_angle

    def get_polyhedral_bond_vectors(self, phase):
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

        # 1) Decide which elements are centers vs. vertices
        centers  = [el for el,info in self.detailed_composition.items() if info.get('polyhedron_center', False)]
        vertices = [el for el,info in self.detailed_composition.items() if info.get('polyhedron_vertex', False)]

        # 2) Initialize empty lists for each center–vertex and each vertex–vertex type
        bond_vectors = {}
        for c in centers:
            for v in vertices:
                bond_vectors[f"{c}-{v}"] = []
        for v1, v2 in combinations_with_replacement(vertices, 2):
            bond_vectors[f"{min(v1,v2)}-{max(v1,v2)}"] = []

        # 3) Build lattice matrix and position maps
        lat = phase.lattice
        a, b, c = lat.a.value, lat.b.value, lat.c.value
        lattice_mat = self._lattice_vectors(a, b, c, lat.alpha.value, lat.beta.value, lat.gamma.value)
        scatterers = phase.getScatterers()
        pos_frac = {i: np.array([atm.x.value, atm.y.value, atm.z.value]) for i, atm in enumerate(scatterers)}
        pos_cart = {i: lattice_mat.dot(frac) for i, frac in pos_frac.items()}
        elements = {i: atm.element for i, atm in enumerate(scatterers)}
        labels   = {i: atm.name.upper() for i, atm in enumerate(scatterers)}

        # 4) Gather which vertices coordinate each center
        coord = {i: [] for i, el in elements.items() if el in centers}

        # 5) Center→vertex bonds using per-center cutoffs
        for i, el in elements.items():
            if el in centers:
                cpos   = pos_cart[i]
                cutoff = self.detailed_composition[el]['cutoff']
                for j, elj in elements.items():
                    if elj in vertices and i != j:
                        vec    = pos_cart[j] - cpos
                        length = np.linalg.norm(vec)
                        if cutoff[0] <= length <= cutoff[1]:
                            rel_len = np.linalg.norm(pos_frac[j] - pos_frac[i])
                            info = {
                                'central_atom': {'symbol': el, 'index': i, 'label': labels[i], 'position': pos_frac[i]},
                                'atom1': {'symbol': el, 'index': i, 'label': labels[i], 'position': pos_frac[i]},
                                'atom2': {'symbol': elj,'index': j, 'label': labels[j], 'position': pos_frac[j]},
                                'vector':         vec,
                                'length':         length,
                                'relative_length':rel_len
                            }
                            bond_vectors[f"{el}-{elj}"].append(info)
                            coord[i].append(j)

        # 6) Vertex–vertex bonds within each polyhedron
        for cen_idx, verts in coord.items():
            cinfo = {'symbol': elements[cen_idx], 'index': cen_idx, 'label': labels[cen_idx], 'position': pos_frac[cen_idx]}
            for u, v in combinations(verts, 2):
                vec     = pos_cart[v] - pos_cart[u]
                length  = np.linalg.norm(vec)
                rel_len = np.linalg.norm(pos_frac[v] - pos_frac[u])
                el_u, el_v = elements[u], elements[v]
                key = f"{min(el_u,el_v)}-{max(el_u,el_v)}"
                edge_info = {
                    'central_atom':   cinfo,
                    'atom1': {'symbol': el_u, 'index': u, 'label': labels[u], 'position': pos_frac[u]},
                    'atom2': {'symbol': el_v, 'index': v, 'label': labels[v], 'position': pos_frac[v]},
                    'vector':         vec,
                    'length':         length,
                    'relative_length':rel_len
                }
                bond_vectors[key].append(edge_info)

        return bond_vectors

    def find_bond_pairs(self, bond_vectors, phase_added_params):
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

    def find_angle_triplets(self, bond_pairs, include_range=(30, 175)):
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
                    angle = self.calculate_angle(vector1, vector2)
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
                        angle1 = self.calculate_angle(vector1 - vector2, vector3 - vector2)
                        if include_range[0] <= angle1 <= include_range[1]:
                            angle_triplets.append({
                                'central_label': central_label, 'atom1_label': atom1_label, 'atom2_label': atom2_label,
                                'atom3_label': atom3_label, 'angle': angle1,
                                'angle name': (atom1_label, atom2_label, atom3_label),
                                'angle category': f"{atom1_label}-{atom2_label}-{atom3_label}"
                            })
                            unique_triplets.add(triplet_sorted)
                    else:
                        angle1 = self.calculate_angle(vector1 - vector2, vector3 - vector2)
                        angle_triplets.append({
                            'central_label': central_label, 'atom1_label': atom1_label, 'atom2_label': atom2_label,
                            'atom3_label': atom3_label, 'angle': angle1,
                            'angle name': (atom1_label, atom2_label, atom3_label),
                            'angle category': f"{atom1_label}-{atom2_label}-{atom3_label}"
                        })
                        unique_triplets.add(triplet_sorted)

        return angle_triplets

    def find_dihedral_quadruplets(self, bond_pairs, angle_threshold=2.0):
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

            # CORRECTED LOGIC: Access the flat dictionary keys directly
            # This aligns with the data structure created by find_bond_pairs
            pos1 = np.array(bond1['atom1_position'])
            pos2 = np.array(bond1['atom2_position'])
            pos3 = np.array(bond2['atom1_position'])
            pos4 = np.array(bond2['atom2_position'])

            # Ensure all four atom positions are unique to form a proper quadruplet
            unique_positions = {tuple(pos1), tuple(pos2), tuple(pos3), tuple(pos4)}
            if len(unique_positions) != 4:
                continue

            # Vectors must be calculated between four distinct points.
            # Here we assume a simple chain A-B-C-D for the calculation.
            # A more robust implementation might test all permutations.
            v1 = pos2 - pos1
            v2 = pos3 - pos2
            v3 = pos4 - pos3

            dihedral_angle = self.calculate_dihedral(v1, v2, v3)
            if dihedral_angle is None or abs(dihedral_angle) < angle_threshold or abs(dihedral_angle) > (180 - angle_threshold):
                continue

            quadruplet = tuple(sorted(list(atoms)))
            if quadruplet in unique_quadruplets:
                continue

            dihedral_quadruplets.append({
                'quadruplet': quadruplet,
                'angle': dihedral_angle,
                'polyhedron_center': polyhedron_center
            })
            unique_quadruplets.add(quadruplet)

        return dihedral_quadruplets

    def detect_edge_bonds(self, bond_vectors, threshold=0.005):
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
            return any(c < threshold or c > 1.0 - threshold for c in pos)

        edge_bonds = {}
        for bond_type, blist in bond_vectors.items():
            if bond_type == 'vertex-vertex':
                continue
            edge_bonds[bond_type] = []
            for bond in blist:
                cpos = bond.get('central_atom', bond['atom1'])['position']
                vpos = bond['atom2']['position']
                if _near_edge(cpos) or _near_edge(vpos):
                    edge_bonds[bond_type].append(bond)

        total = sum(len(v) for v in edge_bonds.values())
        print(f"[INFO] Detected {total} center–vertex bonds near any cell edge.")
        return edge_bonds

    def create_constrain_expressions_for_bonds(self, bond_pairs, phase):
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
            atom1_label, atom2_label = bond['atom1_label'], bond['atom2_label']
            var_name = f'bond_length_{atom1_label}_{atom2_label}_{phase}'
            bond_expr = (
                f"((x_{atom1_label}_{phase} - x_{atom2_label}_{phase})**2 + "
                f"(y_{atom1_label}_{phase} - y_{atom2_label}_{phase})**2 + "
                f"(z_{atom1_label}_{phase} - z_{atom2_label}_{phase})**2)**0.5"
            )
            constrain_dict[var_name] = {
                'atoms': (atom1_label, atom2_label),
                'expression': bond_expr,
                'relative_length': bond['relative_length']
            }
        return constrain_dict

    def create_constrain_expressions_for_angles(self, angle_triplets, phase):
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
            if angle.get('angle') is None or np.isnan(angle['angle']): continue
            atom1_label, central_label, atom2_label = angle['angle name']
            var_name = f'angle_{"_".join(angle["angle name"])}_{phase}'
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
                'angle': angle['angle']
            }
        return constrain_dict

    def create_constrain_expressions_for_dihedrals(self, dihedral_quadruplets, phase):
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


class ResultsManager:
    """
    Handles all result-related tasks: plotting, visualization,
    and saving output files like summaries, data curves, and CIFs.
    """
    def __init__(self, config, analyzer):
        """
        Initializes the ResultsManager.
        ...
        """
        self.config = config
        self.analyzer = analyzer
        self.logger = None

    # ADD THESE TWO METHODS STARTING HERE
    def setup_logging(self, log_file):
        """
        Configures a logger to save progress to a file.

        Args:
            log_file (str): The full path to the log file.
        """
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger('RefinementLogger')
        self.logger.setLevel(logging.INFO)
        # Prevent duplicate handlers if called multiple times
        if not self.logger.handlers:
            # File handler for saving logs to a file
            file_handler = logging.FileHandler(log_file, mode='a')
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Stream handler for printing logs to the console
            stream_handler = logging.StreamHandler()
            stream_formatter = logging.Formatter('%(message)s')
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler)

    def log(self, message, level='info'):
        """
        Writes a message to the configured logger.

        Args:
            message (str): The message to log.
            level (str): The logging level ('info', 'warning', 'error').
        """
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
    
    
    

    def plotmyfit(self, cpdf, baseline=-4, ax=None):
        """
        Plot the observed, calculated, and difference (Gobs, Gcalc, Gdiff) PDFs.

        Parameters:
        - cpdf: PDFContribution object, containing observed profile + model.
        - baseline: Float, baseline offset for difference plot (default: -4).
        - ax: Matplotlib axis object, optional. If None, a new axis is created.

        Returns:
        - fig: The matplotlib Figure object for the plot.
        - df: Pandas DataFrame with columns ['x', 'yobs', 'ycalc', 'ydiff'].
        """
        from matplotlib import pyplot

        if ax is None:
            fig, ax = subplots()
        else:
            fig = ax.get_figure()

        # Pull observed G(r) from cpdf
        x = cpdf.profile.x
        yobs = cpdf.profile.y

        # Pull calculated G(r) from cpdf
        ycalc = cpdf.evaluate()
        ydiff = yobs - ycalc

        ax.plot(
            x, yobs, 'o', label='Gobs',
            markeredgecolor='blue', markerfacecolor='none'
        )
        ax.plot(x, ycalc, color='red', label='Gcalc')
        ax.plot(x, ydiff + baseline, label='Gdiff', color='green')
        ax.plot(x, baseline + 0 * x, linestyle=':', color='black')

        ax.set_xlabel(u'r (Å)')
        ax.set_ylabel(u'G (Å$^{-2}$)')
        ax.legend()

        # Build DataFrame
        df = pd.DataFrame({'x': x, 'yobs': yobs, 'ycalc': ycalc, 'ydiff': ydiff})
        return fig, df

    def visualize_fit_summary(self, cpdf, phase, output_plot_path,
                              font_size=14, label_font_size=20):
        """
        Visualize and summarize the fit results, including PDF data,
        bond lengths, and bond angles.

        Parameters:
        - cpdf: PDFContribution object used in the refinement.
        - phase: Structure object for the current phase.
        - output_plot_path: String, directory path where output files will be saved.
        - font_size: Integer, font size for plot text (default: 14).
        - label_font_size: Integer, font size for axis labels and titles (default: 20).

        Returns:
        - None. Saves summary plots (PDF fit, bond-length histograms, angle histograms).
        """
        import matplotlib.pyplot as plt

        def find_angle_triplets_full(bond_vectors):
            """
            Identify and calculate all center-vertex-vertex and vertex-center-center angles
            based entirely on detailed_composition.
            """
            angle_triplets = []
            unique_triplets = set()
            detailed_composition = self.config.detailed_composition

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
                            angle = self.analyzer.calculate_angle(b1['vector'], b2['vector'])
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
                        angle = self.analyzer.calculate_angle(b1['vector'], b2['vector'])
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

        # 1) Calculate bond vectors (uses the analyzer instance)
        bond_vectors   = self.analyzer.get_polyhedral_bond_vectors(phase)
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

        # ---- Panel 2: Bond-length histograms ----
        detailed_composition = self.config.detailed_composition
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

        # ---- Panel 3: Bond-angle histograms ----
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
        plt.close(fig)

    def evaluate_and_plot(self, cpdf, fitting_range, csv_filename):
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

    def export_cifs(self, i, cpdf, output_dir=None):
            """
            Export the refined structures from the PDFContribution object (`cpdf`) to CIF files.
    
            Parameters:
            - i: Integer, the step or iteration index used in naming the output files.
            - cpdf: PDFContribution object, containing the refined structural models for each phase.
            - output_dir (str, optional): The directory to save the CIFs. Defaults to
              the standard output path if not provided.
            """
            # Use the provided output_dir, or fall back to the default project path.
            output_path = output_dir if output_dir is not None else self.config.output_results_path
            for phase in cpdf._generators:
                getattr(cpdf, str(phase)).stru.write(
                    os.path.join(output_path, f"{phase}_{i}.cif"), format='cif'
                )

    def saveResults(self, i, fit, cpdf):
        """
        Save the results of a refinement step, including fitting summary, visualizations,
        and data files.

        Parameters:
        - i: Integer, the step or iteration index used in naming the output files.
        - fit: FitRecipe object, containing the refined model and fit results.
        - cpdf: PDFContribution object, containing the same contribution that was used
                to build/refine `fit`.  This is needed for both plotmyfit() and export_cifs().
        """
        from matplotlib.pyplot import close
        output_results_path = self.config.output_results_path

        # Remove any constraint whose .par is None
        fit._oconstraints[:] = [c for c in fit._oconstraints if c.par is not None]

        # Create a FitResults summary and write it to disk
        res = FitResults(fit)
        res.saveResults(os.path.join(output_results_path, f'fitting_summary_{i}.txt'))

        # Plot and save the “fit vs. data” figure, passing cpdf explicitly
        fig0, df = self.plotmyfit(cpdf)
        fig0.savefig(os.path.join(output_results_path, f'fitting_{i}.png'), dpi=600)
        close(fig0)

        # Save G(r), G_calc, and G_diff to CSV
        df.to_csv(os.path.join(output_results_path, f'fitting_curve_{i}.csv'), index=False)

        # Export all phases to CIF files
        self.export_cifs(i, cpdf)

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

    def finalize_results(self, cpdf, fit):
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
        from matplotlib.pyplot import close
        output_results_path = self.config.output_results_path
        myrange = self.config.myrange
        myrstep = self.config.myrstep
        
        # Export partial PDFs of each phase
        if len(cpdf._generators) > 1:
            scale_factors = {}

            # Read out the scale factors
            for phase in cpdf._generators:
                scale_factors[phase] = getattr(cpdf, 's_' + str(phase)).value

            # Zero out the scale factors
            for phase in cpdf._generators:
                getattr(cpdf, 's_' + str(phase)).value = 0

            # Generate a partial PDF for each phase
            for phase in scale_factors:
                getattr(cpdf, 's_' + str(phase)).value = scale_factors[phase]
                fig0, df = self.plotmyfit(cpdf)
                fig0.savefig(os.path.join(output_results_path, f'{phase}_partial_fit.png'), dpi=600)
                close(fig0)
                df.to_csv(os.path.join(output_results_path, f'{phase}_partial.csv'), index=False)
                getattr(cpdf, 's_' + str(phase)).value = 0

            # Restore the original scale factors
            for phase in scale_factors:
                getattr(cpdf, 's_' + str(phase)).value = scale_factors[phase]

        # Extrapolation of the fit to a full range regardless of the fitting steps
        cpdf.setCalculationRange(myrange[0], myrange[1], myrstep)
        fig0, df = self.plotmyfit(cpdf)
        fig0.savefig(os.path.join(output_results_path, '_final_extrapolated_fit.png'), dpi=600)
        close(fig0)
        df.to_csv(os.path.join(output_results_path, '_final_extrapolated_fit.csv'), index=False)

        print(f"Results finalized and saved to {output_results_path}")
        
 

    def log_and_plot_rw(self, output_path, stage_index, iteration_history):
        """
        Logs the Rw value for each iteration to a CSV file and generates a
        convergence plot. Both files are overwritten at each call to reflect
        the latest state.

        Args:
            output_path (str): The directory to save the log and plot.
            stage_index (int): The current refinement stage index (i).
            iteration_history (list): A list of tuples, where each tuple
                                      contains (iteration_number, Rw_value).
        """

        # Ensure the history is not empty
        if not iteration_history:
            return

        # Define file paths
        log_filename = os.path.join(output_path, f'rw_log_{stage_index}.csv')
        plot_filename = os.path.join(output_path, f'rw_plot_{stage_index}.png')

        # --- 1. Update Log File ---
        try:
            # Create a DataFrame and save to CSV, overwriting the file
            df = pd.DataFrame(iteration_history, columns=['Iteration', 'Rw'])
            df.to_csv(log_filename, index=False)
        except Exception as e:
            print(f"[ERROR] Could not write to log file {log_filename}: {e}")
            exit # Exit if logging fails
    
    
    
    def save_state_callback(self, fit, cpdf, output_path, lock, iteration_counter, save_every=1):
        """
        Callback to save the current state of the FitRecipe object and CIFs.
    
        This method uses dill for robust serialization and a threading lock to
        ensure file operations are thread-safe. It overwrites previous saves
        to keep only the most recent intermediate state.
    
        Args:
            fit (FitRecipe): The current FitRecipe object.
            cpdf (PDFContribution): The current PDFContribution object.
            output_path (str): The directory to save the files.
            lock (threading.Lock): The lock to ensure thread-safe file writing.
            iteration_counter (int): The current iteration number.
            save_every (int): The frequency (in iterations) to save the state.
        """
        
        # This functionality is currently disabled but can be re-enabled for
        # long refinements.
        
        # Save only on the specified interval.
        # if iteration_counter > 0 and iteration_counter % save_every == 0:
        #     print(f"Saving current state at iteration {iteration_counter}...")
            
        #     # Acquire the lock to ensure thread-safe file operations.
        #     with lock:
        #         # Define a fixed filename to overwrite the previous state.
        #         dill_filename = os.path.join(output_path, 'fit_state_current.dill')
        #         try:
        #             with open(dill_filename, 'wb') as f:
        #                 dill.dump(fit, f)
        #         except Exception as e:
        #             print(f"[ERROR] Could not save dill file: {e}")
                
        #         # Export the current refined structures to CIFs with a fixed name.
        #         try:
        #             # Use a 'current' tag to indicate this is the latest intermediate file.
        #             self.export_cifs('current', cpdf)
        #         except Exception as e:
        #             print(f"[ERROR] Could not save CIF file: {e}")
            
        #     print(f"Current state saved to {dill_filename} and CIFs exported.")




class PDFManager:
    """
    Handles the creation and management of PDF data and contributions.
    This class encapsulates reading experimental data to generate a PDF (G(r))
    and building PDFContribution objects from structural models (CIF files).
    """
    def __init__(self, config, ncpu, pool):
        """
        Initializes the PDFManager.

        Parameters:
        - config: A configuration object containing paths and parameters like
                  qdamp, qbroad, cif_directory, etc.
        - ncpu: The number of CPUs to use for parallel processing.
        - pool: The multiprocessing pool instance.
        """
        self.config = config
        self.ncpu = ncpu
        self.pool = pool


    def generatePDF(self, data_directory, data_filename, composition, qmax, myrange, myrstep, pdfgetx_config):
        """
        Generates a Pair Distribution Function (G(r)) from powder diffraction data.

        Args:
            data_directory (str): Path to the directory containing the data file.
            data_filename (str): The name of the powder diffraction data file.
            composition (str): Atomic composition of the sample (e.g., "O7 V2 Zr1").
            qmax (float): Maximum Q value for the Fourier transform.
            myrange (tuple): The (rmin, rmax) range for the output PDF.
            myrstep (float): The step size (Δr) for the output PDF.
            pdfgetx_config (dict): Dictionary with settings for the PDFGetter instance.

        Returns:
            tuple: A tuple containing the following three elements:
                r0 (np.ndarray): The r-values of the calculated PDF.
                g0 (np.ndarray): The G(r) values of the calculated PDF.
                cfg (PDFConfig): The PDFConfig object used for the calculation.
        """
        # 1. Construct the full file path internally.
        filepath = os.path.join(data_directory, data_filename)

        # 2. Set up the PDF configuration by combining general and specific settings.
        cfg_params = pdfgetx_config.copy()
        cfg_params.update({
            'composition': composition,
            'rstep': myrstep,
            'rmin': myrange[0],
            'rmax': myrange[1]
        })

        # Create the PDFConfig object by unpacking the complete dictionary.
        cfg = PDFConfig(**cfg_params)

        # Set qmax separately as it's used for both qmax and qmaxinst.
        cfg.qmax = cfg.qmaxinst = qmax

        # 3. Generate the PDF using the constructed data file path.
        pg0 = PDFGetter(config=cfg)
        r0, g0 = pg0(filename=filepath)

        return r0, g0, pg0.config


    def contribution(self, name, qmin, qmax, ciffile, periodic, super_cell):
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

        # allow an expansion to a supercell, which is defined by a 3-element tuple
        structure = expansion.supercell(structure, (super_cell[0], super_cell[1], super_cell[2]))
        pdfgenerator.setStructure(structure, periodic=periodic)
        return pdfgenerator

    def DebyeContribution(self, name, qmin, qmax, ciffile, periodic, super_cell):
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
        pdfgenerator = DebyePDFGenerator(str(name))
        pdfgenerator.setQmax(qmax)
        pdfgenerator.setQmin(qmin)
        pdfgenerator._calc.evaluatortype = 'OPTIMIZED'
        structure = loadStructure(ciffile)

        # allow an expansion to a supercell, which is defined by a 3-element tuple
        structure = expansion.supercell(structure, (super_cell[0], super_cell[1], super_cell[2]))
        pdfgenerator.setStructure(structure, periodic=periodic)
        return pdfgenerator

    def build_contribution(self, r0, g0, cfg, cif_info, fitRange, name='cpdf'):
        """
        Create a PDFContribution object for the refinement process by combining experimental data
        with structure models from CIF files. This method is the OOP equivalent of the
        original 'phase' function.

        Parameters:
        - r0: Numpy array, the observed r values of the PDF data.
        - g0: Numpy array, the observed g values of the PDF data.
        - cfg: PDFConfig object, containing the configuration for PDF generation.
        - cif_info: Dictionary, where keys are CIF filenames and values are lists containing
          [space group, periodicity (True/False), supercell dimensions (tuple)].
        - fitRange: Tuple of two floats, the r range for the fitting (xmin, xmax).
        - name: A string name for the PDFContribution object.

        Returns:
        - cpdf: PDFContribution object, representing the combined phases with associated constraints
          and experimental data.
        """
        cpdf = PDFContribution(name)
        cpdf.setScatteringType('X')
        cpdf.setQmax(cfg.qmax)
        cpdf.profile.setObservedProfile(r0.copy(), g0.copy())
        cpdf.setCalculationRange(xmin=fitRange[0], xmax=fitRange[1], dx=self.config.myrstep)

        # specify an independent structure model
        for i, file in enumerate(cif_info.keys()):
            print('Phase: ', i, file)
            periodic = list(cif_info.values())[i][1]
            super_cell = list(cif_info.values())[i][2]
            print('Phase periodic? ', periodic)
            
            cif_path = os.path.join(self.config.cif_directory, file)
            pdf = self.contribution('Phase' + str(i), cfg.qmin, cfg.qmax, cif_path, periodic, super_cell)
            
            pdf.qdamp.value = self.config.qdamp
            pdf.qbroad.value = self.config.qbroad
            pdf.parallel(ncpu=self.ncpu, mapfunc=self.pool.map)
            cpdf.addProfileGenerator(pdf)

        cpdf.setResidualEquation('resv')
        return cpdf
    
    
class RefinementHelper:
    """
    A utility class for housing various helper functions used throughout the
    refinement workflow. These methods perform common, reusable tasks.
    """

    @staticmethod
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
    

class PDFRefinement:
    """
    Main class to orchestrate the entire PDF refinement workflow. It manages
    the configuration, data, contribution, recipe objects, and execution flow.
    """
    def __init__(self, config, pdf_manager, results_manager, helper, analyzer, ncpu, pool):
        self.config = config
        self.pdf_manager = pdf_manager
        self.results_manager = results_manager
        self.helper = helper
        self.analyzer = analyzer
        self.ncpu = ncpu
        self.pool = pool 
        self.cpdf = None
        self.fit = None
        self.added_params = {}
        self.global_bond_vectors = {}
        self.iteration_counter = 0
        self.file_lock = threading.Lock()
        self.total_iteration_count = 0
    
    def log(self, message, level='info'):
        """Convenience method to call the ResultsManager logger."""
        self.results_manager.log(message, level)

    def run_refinement_step(self, i, fitting_range, myrstep, fitting_order, residualEquation, save_every=1):
        """
        Perform the refinement process for a single stage index "i".
        
        Parameters:
        - i: Integer, index of the fitting step.
        - fitting_range: List of two floats, the r range for the fit (e.g., [1.5, 27]).
        - myrstep: Float, step size for r values.
        - fitting_order: List of strings, tags defining the fitting order (e.g., ['lat', 'scale', 'xyz']).
        - residualEquation: String, residual equation used for fitting (e.g., 'resv').
        - save_very: frequency of progress logging, default = 1 (each iteration)

        Returns:
        - None

        """
        print('-------------------------------------------------------------------')
        print(f"Fitting stage {i}")
        print('-------------------------------------------------------------------')
        
        self.cpdf.setCalculationRange(fitting_range[0], fitting_range[1], myrstep)
        self.cpdf.setResidualEquation(residualEquation)
        self.fit.fix('all')
        
        # Reset the iteration counter and history for this new refinement stage.
        self.iteration_counter = 0
        iteration_history = [] # Stores (iteration, Rw) tuples

        def callback(xk):
            """
            Nested callback to manage iteration state, save intermediate
            results, calculate Rw, and log/plot convergence.
            """
            self.iteration_counter += 1

            # --- Calculate Rw for the current state ---
            try:
                r = self.cpdf.profile.x
                g_obs = self.cpdf.profile.y
                g_calc = self.cpdf.evaluate() # Get G_calc at current state

                # Mask to the fitting range for accurate Rw
                rmin, rmax = fitting_range
                mask = (r >= rmin) & (r <= rmax)
                g_obs_fit = g_obs[mask]
                g_calc_fit = g_calc[mask]

                # Calculate Rw
                numerator = np.sum((g_obs_fit - g_calc_fit)**2)
                denominator = np.sum(g_obs_fit**2)
                rw = np.sqrt(numerator / denominator) if denominator != 0 else 0

                # Append to history
                iteration_history.append((self.iteration_counter, rw))

                # Log and plot the updated history
                self.results_manager.log_and_plot_rw(
                    output_path=self.config.output_results_path,
                    stage_index=i,
                    iteration_history=iteration_history
                )
            except Exception as e:
                print(f"[WARNING] Could not calculate or log Rw at iteration {self.iteration_counter}: {e}")

            # --- Save intermediate state (CIF object) ---
            self.results_manager.save_state_callback(
                fit=self.fit,
                cpdf=self.cpdf,
                output_path=self.config.output_results_path,
                lock=self.file_lock,
                iteration_counter=self.iteration_counter,
                save_every=save_every
            )
    
        for step in fitting_order:
            try:
                print(f'\nFreeing parameter group: {step}')
                self.fit.free(step)
                optimizer = 'L-BFGS-B'
                minimize(self.fit.scalarResidual, self.fit.values, method=optimizer, 
                         options=self.config.convergence_options, callback=callback)
            except Exception as e:
                print(f"Caught an exception during minimization for step '{step}': {e}")
                continue
    
        # Save the final results for this stage.
        self.results_manager.saveResults(i, self.fit, self.cpdf)
    
        # Generate and save the final statistics plots for this stage.
        for phase_name in self.cpdf._generators:
            phase = getattr(self.cpdf, str(phase_name)).phase
            output_path_prefix = os.path.join(self.config.output_results_path, f"{i}_{phase_name}_")
            self.results_manager.visualize_fit_summary(self.cpdf, phase, output_path_prefix)


            
    def build_initial_recipe(self):
        """Constructs the initial FitRecipe object for the refinement.

        It systematically defines and adds all necessary
        parameters, applies symmetry constraints, and prepares the model for
        the first optimization step.
        
        The process involves the following sequential operations:
        
        1.  **Recipe Initialization**:
            - A new, empty `FitRecipe` object is instantiated and assigned to
              `self.fit`.
            - The `PDFContribution` object, containing the experimental data
              and structural model generators, is added to the recipe.
        
        2.  **Core Parameter Definition**:
            - **Scale Factors**: A scale factor variable (`s_Phase...`) is
              created for each structural phase. This parameter accounts for
              the phase fraction and the overall experimental intensity scale.
            - **Correlated Motion (delta2)**: A `delta2` variable is added for
              each phase. This parameter models the sample-dependent peak
              sharpening at low-r that arises from correlated atomic motion,
              a phenomenon not described by the standard atomic displacement
              parameters (ADPs).
        
        3.  **Crystallographic Parameter Generation**:
            - **Symmetry Application**: The `constrainAsSpaceGroup` function is
              invoked to analyze the crystal structure of each phase. It
              determines the set of independent refinable parameters (lattice,
              ADP, and atomic coordinates) consistent with the specified
              space group symmetry.
            - **Lattice Parameters**: The independent lattice parameters (e.g.,
              'a', 'b', 'c', 'alpha', 'beta', 'gamma') identified by the
              symmetry analysis are added to the recipe as refinable variables.
            - **Atomic Displacement Parameters (ADPs)**: Parameters describing
              thermal and static atomic disorder are added. The method follows
              the configuration setting:
              - If `anisotropic` is True, independent anisotropic displacement
                parameters (`U11`, `U22`, etc.) are added.
              - If `anisotropic` is False, isotropic parameters (`Uiso`) are
                added. If `unified_Uiso` is True, all atoms of the same
                element within a phase are constrained to share a single
                `Uiso` value.
            - **Atomic Coordinates**: The independent fractional atomic
              coordinates ('x', 'y', 'z') determined by the symmetry analysis
              are added to the recipe as refinable variables.
        
        4.  **Nanoparticle Shape Function**:
            - The `sphericalCF` characteristic function is registered for each
              phase to model the damping of the PDF signal that results from
              finite crystallite size.
            - A corresponding `psize` parameter, representing the average
              diameter of the spherical nanoparticles, is added for each phase.
            - The overall phase equation is updated to include this shape
              function multiplicative term.
        
        5.  **Initial Bond Vector Calculation**:
            - As a final preparatory step, the `get_polyhedral_bond_vectors`
              method is called to calculate all bond lengths and angles from
              the initial, unrefined structure.
            - These geometric descriptors are stored in the instance variable
              `self.global_bond_vectors` for later use when applying rigid-body
              constraints via the `apply_rigid_body_constraints` method.
        
        Returns:
            diffpy.srfit.fitbase.FitRecipe: The fully constructed and configured
            `FitRecipe` object, ready for the first refinement step.
        """
        self.fit = FitRecipe()
        self.fit.addContribution(self.cpdf)

        #------------------ 1 ------------------#
        # Generate a phase equation.
        phase_equation = ' + '.join([f's_{phase}*{phase}' for phase in self.cpdf._generators])
        self.cpdf.setEquation(phase_equation)
        print('equation:', self.cpdf.getEquation())

        #------------------ 2b ------------------#
        # add delta2
        for phase in self.cpdf._generators:
            self.fit.addVar(getattr(self.cpdf, phase).delta2, name=f'delta2_{phase}', value=2.0,
                            tags=['delta2', str(phase), f'delta2_{phase}', 'delta'])
            self.fit.restrain(getattr(self.cpdf, phase).delta2, lb=0.0, ub=5, scaled=True, sig=0.005)

        #------------------ 4 ------------------#
        # add scale factors s*
        for phase in self.cpdf._generators:
            self.fit.addVar(getattr(self.cpdf, f's_{phase}'), value=0.1, tags=['scale', str(phase), f's_{phase}'])
            self.fit.restrain(getattr(self.cpdf, f's_{phase}'), lb=0.0, scaled=True, sig=0.0005)

        #------------------ 5 ------------------#
        # determine independent parameters based on the space group
        for i, phase in enumerate(self.cpdf._generators):
            spaceGroup = str(list(self.config.ciffile.values())[i][0])
            sgpar = constrainAsSpaceGroup(getattr(self.cpdf, phase).phase, spaceGroup, sgoffset=self.config.sgoffset)

            #------------------ 6 ------------------#
            # add lattice parameters
            for par in sgpar.latpars:
                self.fit.addVar(par, value=par.value, name=f'{par.name}_{phase}', fixed=False, tags=['lat', str(phase), f'lat_{phase}'])

            #------------------ 7a ------------------#
            # atomic displacement parameters ADPs
            if self.config.anisotropic:
                getattr(self.cpdf, phase).stru.anisotropy = True
                print('Adding anisotropic displacement parameters.')
                for par in sgpar.adppars:
                    atom = par.par.obj
                    atom_label = atom.label
                    atom_symbol = atom.element
                    # take Uiso initial from detailed_composition
                    u0 = self.config.detailed_composition.get(atom_symbol, {}).get('Uiso', 0.01)
                    name = f"{par.name}_{atom_label}_{phase}"
                    tags = ['adp', f"adp_{atom_label}", f"adp_{atom_symbol}_{phase}", f"adp_{phase}", f"adp_{atom_symbol}", str(phase)]
                    self.fit.addVar(par, value=u0, name=name, tags=tags)
                    self.fit.restrain(par, lb=0.0, ub=0.1, scaled=True, sig=0.0005)
            else:
                mapped_adppars = self.helper.map_sgpar_params(sgpar, 'adppars')
                added_adps = set()
                for par in sgpar.adppars:
                    try:
                        atom_symbol = par.par.obj.element
                        parameter_name = par.name
                        atom_label = mapped_adppars[parameter_name][1]
                        added_adps.add(atom_label)
                    except Exception:
                        pass
                if self.config.unified_Uiso:
                    print('Adding isotropic displacement parameters as unified values.')
                    getattr(self.cpdf, phase).stru.anisotropy = False
                    # one Uiso per element
                    for el, info in self.config.detailed_composition.items():
                        u0 = info['Uiso']
                        var = self.fit.newVar(f"Uiso_{el}_{phase}", value=u0, tags=['adp', el, str(phase)])
                        for atom in getattr(self.cpdf, phase).phase.getScatterers():
                            if atom.element == el:
                                self.fit.constrain(atom.Uiso, var)
                                self.fit.restrain(atom.Uiso, lb=0.0, ub=0.1, scaled=True, sig=0.0005)
                else:
                    print('Adding isotropic displacement parameters as independent values.')
                    getattr(self.cpdf, phase).stru.anisotropy = False
                    for atom in getattr(self.cpdf, phase).phase.getScatterers():
                        el = atom.element
                        if atom.name.upper() in added_adps:
                            u0 = self.config.detailed_composition.get(el, {}).get('Uiso', 0.01)
                            self.fit.addVar(atom.Uiso,
                                       value=u0,
                                       name=f"{atom.name}_{phase}",
                                       fixed=False,
                                       tags=['adp', str(phase), f"adp_{phase}", f"adp_{el}"])
                            self.fit.restrain(atom.Uiso, lb=0.0, ub=0.1, scaled=True, sig=0.0005)



            #------------------ 8 ------------------#
            # atom positions XYZ
            self.added_params[str(phase)] = set()
            mapped_xyzpars = self.helper.map_sgpar_params(sgpar, 'xyzpars')
            for par in sgpar.xyzpars:
                try:
                    atom_symbol = par.par.obj.element
                    p_name = par.name
                    mapped_name, atom_label = mapped_xyzpars[p_name]
                    name_long = f"{mapped_name}_{phase}"
                    tags = ['xyz', f'xyz_{atom_symbol}', f'xyz_{atom_symbol}_{phase}', f'xyz_{phase}', str(phase)]
                    self.fit.addVar(par, name=name_long, tags=tags)
                    self.added_params[str(phase)].add(atom_label)
                    print(f"Constrained {name_long} at {par.par.value}: atom position added to variables.")
                except Exception:
                    pass

        #------------------ 10 ------------------#
        for phase in self.cpdf._generators:
            self.cpdf.registerFunction(sphericalCF, name=f'sphere_{phase}', argnames=['r', f'psize_{phase}'])
        
        phase_equation_scf = ' + '.join([f's_{phase}*{phase}*sphere_{phase}' for phase in self.cpdf._generators])
        self.cpdf.setEquation(phase_equation_scf)
        print('equation:', self.cpdf.getEquation())

        for phase in self.cpdf._generators:
            self.fit.addVar(getattr(self.cpdf, f'psize_{phase}'), value=100.0, fixed=False, tags=['psize', str(phase), f'psize_{phase}'])
            self.fit.restrain(getattr(self.cpdf, f'psize_{phase}'), lb=0.0, scaled=True, sig=0.1)

        # show diagnostic information
        self.fit.fithooks[0].verbose = 2

        #------------------ 11 ------------------#
        for phase in self.cpdf._generators:
            print(f"Calculating bond vectors for {phase}")
            bond_vectors = self.analyzer.get_polyhedral_bond_vectors(getattr(self.cpdf, phase).phase)
            self.global_bond_vectors[str(phase)] = bond_vectors

        return self.fit

    def modify_recipe_spacegroup(self, spacegroup_list):
        """
        Modifies the FitRecipe to apply a new space group symmetry.
        
        It rebuilds the atomic parameter set of the `FitRecipe` to be
        consistent with a new space group while preserving the refined atomic
        positions and lattice constants as much as possible.
    
        The process involves the following sequential operations:
    
        1.  **State Preservation**:
            - The current refined values of all atomic coordinate variables
              (names starting with 'x_', 'y_', 'z_') are extracted from the
              `FitRecipe` and stored in a temporary dictionary.
    
        2.  **Variable Purging**:
            - All existing atomic coordinate variables are removed from the
              `FitRecipe`.
            - All existing rigid-body constraint variables (names starting
              with 'bond_', 'angle_', 'dihedral_') are also removed. This
              is a necessary step because the set of independent atoms and
              their geometric relationships will change with the new symmetry,
              rendering the previous constraints invalid.
    
        3.  **New Symmetry Application**:
            - The `constrainAsSpaceGroup` function is called for each phase
              using the new space group string provided in `spacegroup_list`.
            - This operation re-analyzes the structure to determine the new set
              of independent atomic coordinates required to describe the model
              under the new symmetry constraints.
    
        4.  **Parameter Regeneration**:
            - The method iterates through the newly determined independent
              atomic position parameters.
            - For each parameter, it adds a new variable to the `FitRecipe`.
              The initial value of this new variable is set from the preserved
              coordinates if a matching atom label exists, thus ensuring a
              continuous and physically reasonable structural model across the
              symmetry change.
    
        5.  **Lattice Parameter Handling**:
            - The method re-applies constraints to the lattice parameters. It
              specifically uses the original high-symmetry space group from the
              project configuration to do this. This enforces a specific
              lattice geometry (e.g., pseudo-cubic) even when the atomic
              arrangement conforms to a lower symmetry.
    
        Args:
            spacegroup_list (list): A list of space group strings (e.g.,
                ['P213', 'P1']), one for each phase in the model.
    
        Returns:
            diffpy.srfit.fitbase.FitRecipe: The modified `FitRecipe` object,
            ready for further refinement under the new symmetry constraints.
        """
        # Step 1: Read out all atom position variables from the fit
        old_xyz_vars = {name: getattr(self.fit, name).value for name in self.fit.names if name.startswith(('x_', 'y_', 'z_'))}
        
        # Steps 3 & 4: Delete old variables
        vars_to_delete = [name for name in self.fit.names if name.startswith(('x_','y_','z_','bond_','angle_','dihedral_'))]
        for name in vars_to_delete:
            try:
                self.fit.unconstrain(getattr(self.fit, name))
                self.fit.delVar(getattr(self.fit, name))
                print(f"{name}: old variable deleted")
            except (AttributeError, KeyError, ValueError):
                pass
        
        for phase in self.cpdf._generators:
            self.added_params[str(phase)] = set()

        # Step 5: Apply the new space group and generate new variables
        for i, phase in enumerate(self.cpdf._generators):
            try:
                sgpar = constrainAsSpaceGroup(getattr(self.cpdf, phase).phase, spacegroup_list[i])
                sgpar._clearConstraints() # Crucial step 
                
                mapped_xyzpars = self.helper.map_sgpar_params(sgpar, 'xyzpars')
                for par in sgpar.xyzpars:
                    try:
                        name_long = f"{mapped_xyzpars[par.name][0]}_{phase}"
                        old_value = old_xyz_vars.get(name_long, par.par.value)
                        tags = ['xyz', str(phase)] # Simplified tags
                        self.fit.addVar(par, value=old_value, name=name_long, tags=tags)
                        self.added_params[str(phase)].add(mapped_xyzpars[par.name][1])
                        print(f"Constrained {name_long} at {old_value}: atom position added to variables.")
                    except Exception:
                        pass # Ignore if parameter already exists
            except Exception as e:
                print(f"Error applying space group to phase {phase}: {e}")

        # Step 6: Enforce Pseudo-Cubic Constraints for Lattice Parameters and anisotropic ADPs
        old_lattice_vars = {}
        for name in self.fit.names:
            if name.startswith(('a_', 'b_', 'c_', 'alpha_', 'beta_', 'gamma_')):
                var_value = getattr(self.fit, name).value
                old_lattice_vars[name] = var_value
    
        for name in old_lattice_vars.keys():
            try:
                self.fit.unconstrain(getattr(self.fit, name))
                # fit.clearConstraints(getattr(fit, name))
                # fit.clearRestraints(getattr(fit, name))
                self.fit.delVar(getattr(self.fit, name))
                print(f"{name}: old variable deleted")
            except Exception:
                pass
    
        for phase in self.cpdf._generators:
            spaceGroup = str(list(self.config.ciffile.values())[0][0])
            sgpar = constrainAsSpaceGroup(getattr(self.cpdf, phase).phase, spaceGroup, sgoffset=self.config.sgoffset)
            for par in sgpar.latpars:
                name = par.name + '_' + str(phase)
                try:
                    old_value = old_lattice_vars[name]
                    self.fit.addVar(par, value=old_value, name=name, fixed=False, tags=['lat', str(phase), 'lat_' + str(phase)])
                    print(f"Constrained {name} at {old_value}.")
                except Exception:
                    pass
    
        # strip out any constraint whose .par is None
        self.fit._oconstraints[:] = [c for c in self.fit._oconstraints if c.par is not None]




        return self.fit   
    
# In sample_refinement_v21_classes.py, inside the PDFRefinement class:

    def update_recipe_from_initial(self,
            fit_old,
            fit_new,
            cpdf_new,
            recalculate_bond_vectors=False):
        """Updates a new FitRecipe with refined values from a previous fit.

       This method takes the results from a completed refinement (`fit_old`) and uses them
       to initialize the parameters for the next refinement stage (`fit_new`). This
       ensures continuity, provides a better starting point for the optimizer,
       and typically leads to faster and more stable convergence.
    
       The core functionality involves copying the `.value` of each parameter
       from the old recipe to the new one. It also provides an option to
       recalculate derived structural properties (e.g., bond vectors) that may
       have become outdated due to changes in the structure during the previous
       refinement.
    
       Args:
           fit_old (FitRecipe):
               The source `FitRecipe` object from a *completed* refinement. Its
               parameter values will be used as the source for initialization.
           fit_new (FitRecipe):
               The target `FitRecipe` object that will be initialized for the
               *next* refinement step. This object is modified in place.
           cpdf_new (PDFContribution):
               The `PDFContribution` object associated with `fit_new`. This is
               essential if bond vectors need recalculation, as it holds the
               up-to-date structural phase information.
           recalculate_bond_vectors (bool, optional):
               If True, recalculates structural properties like polyhedral bond
               vectors using the structure from `cpdf_new`. This should be enabled
               when preceding refinement steps have altered lattice or atomic
               parameters. Defaults to False.
    
       Returns:
           FitRecipe:
               The updated `fit_new` object, now populated with values from
               `fit_old` and set as the active recipe for the workflow via
               `self.fit`.
   """
        print("\n======================================")
        print("[INFO] Updating new recipe with initial values from previous fit")
        print("======================================\n")

        # Copy values from the old fit to the new one
        print("[INFO] Copying refined values from previous fit into the new recipe...")
        for name in fit_old.names:
            if name in fit_new.names:
                try:
                    old_val = getattr(fit_old, name).value
                    getattr(fit_new, name).value = old_val
                    # print(f"[INFO]   Copied '{name}' = {old_val}") # For debugging
                except AttributeError:
                    print(f"[WARNING] Could not copy value for '{name}'.")
        print("[INFO] Value copy complete.")

        if recalculate_bond_vectors:
            # CORRECTED: Use the cpdf_new object passed into the function
            for phase in cpdf_new._generators:
                phase_generator = getattr(cpdf_new, phase)
                bond_vectors = self.analyzer.get_polyhedral_bond_vectors(phase_generator.phase)
                self.global_bond_vectors[phase] = bond_vectors
                print(f"[INFO] Recalculated bond vectors for phase '{phase}'")

        # Final setup
        if fit_new.fithooks:
            fit_new.fithooks[0].verbose = 2

        # Update the workflow's main fit object to the new one
        self.fit = fit_new

        return self.fit
 

    
    def apply_rigid_body_constraints(self, constrain_bonds, constrain_angles, constrain_dihedrals, adaptive=False):
        """
        Applies geometric restraints to bonds, angles, and dihedrals.
        
        This method enhances a structural refinement by applying soft restraints
        to maintain chemically plausible geometries. Rather than enforcing rigid
        constraints, it adds a penalty to the fit's cost function if bond
        lengths, angles, or dihedrals deviate from their expected values. This
        is essential for stabilizing refinements of complex, disordered, or
        low-resolution structures where atomic positions might otherwise drift
        into unrealistic configurations.
        
        The function dynamically identifies bonds, angles, and dihedrals, creates
        corresponding variables in the `FitRecipe`, and then applies a `restrain`
        call with specified bounds and a sigma value, which controls the
        "stiffness" of the restraint.
        
        Args:
        constrain_bonds (tuple[bool, float]):
            A tuple to control bond restraints. The first element is a
            **boolean** to enable or disable them. The second is a **float**
            specifying the default `sigma` (standard deviation), which
            determines how strongly the bond lengths are restrained. A
            smaller sigma means a stronger restraint.
        constrain_angles (tuple[bool, float]):
            A tuple to control angle restraints. The first element is a
            **boolean** to enable or disable them. The second is a **float**
            for the `sigma` value applied to all bond angle restraints.
        constrain_dihedrals (tuple[bool, float]):
            A tuple to control dihedral angle restraints. The first element
            is a **boolean** to enable or disable them. The second is a **float**
            for the `sigma` value applied to all dihedral restraints.
        adaptive (bool, optional):
            If `True`, the underlying bond vectors and geometries are
            recalculated at the start of this function based on the current
            state of the structure in the recipe. If `False` (default), it
            uses a pre-calculated, globally stored set of bond vectors.
        
        Returns:
            diffpy.srfit.fit.FitRecipe:
                The updated `FitRecipe` object containing the newly added
                restraint variables and expressions.
        """
        
        print("\n--------------------------------------")
        print("[INFO] Adding constraints on bonds, angles, and dihedrals")
        print("--------------------------------------")

        for i, phase in enumerate(self.cpdf._generators):
            phase_key = str(phase)

            # Retrieve or recalculate bond vectors using the analyzer
            if not adaptive:
                bond_vectors = self.global_bond_vectors[phase_key]
            else:
                bond_vectors = self.analyzer.get_polyhedral_bond_vectors(getattr(self.cpdf, phase).phase)

            # Use the analyzer for all geometric calculations
            edge_bonds = self.analyzer.detect_edge_bonds(bond_vectors, threshold=0.005)
            bond_pairs = self.analyzer.find_bond_pairs(bond_vectors, self.added_params[phase_key])
            angle_triplets = self.analyzer.find_angle_triplets(bond_pairs)
            dihedral_quadruplets = self.analyzer.find_dihedral_quadruplets(bond_pairs)

            # Use the analyzer to generate constraint expressions
            constrain_dict_bonds = self.analyzer.create_constrain_expressions_for_bonds(bond_pairs, phase_key)
            constrain_dict_angles = self.analyzer.create_constrain_expressions_for_angles(angle_triplets, phase_key)
            constrain_dict_dihedrals = self.analyzer.create_constrain_expressions_for_dihedrals(dihedral_quadruplets, phase_key)

            # --------------------------------------------------------------------
            # BOND CONSTRAINTS
            # --------------------------------------------------------------------
            if constrain_bonds[0]:
                print("\n[INFO] Processing bond constraints...")
                # This complex logic is preserved from the original script
                shared_vertex_dict = {}
                centers  = [el for el,info in self.config.detailed_composition.items() if info.get('polyhedron_center')]
                vertices = [el for el,info in self.config.detailed_composition.items() if info.get('polyhedron_vertex')]

                for var_name, data in constrain_dict_bonds.items():
                    atom1_label, atom2_label = data['atoms']
                    center_lbl, vertex_lbl = None, None
                    if any(atom1_label.startswith(c) for c in centers) and any(atom2_label.startswith(v) for v in vertices):
                        center_lbl, vertex_lbl = atom1_label, atom2_label
                    elif any(atom2_label.startswith(c) for c in centers) and any(atom1_label.startswith(v) for v in vertices):
                        center_lbl, vertex_lbl = atom2_label, atom1_label
                    if vertex_lbl:
                        shared_vertex_dict.setdefault(vertex_lbl, {'bond_vars': [], 'expressions': []})
                        shared_vertex_dict[vertex_lbl]['bond_vars'].append(var_name)
                        shared_vertex_dict[vertex_lbl]['expressions'].append(data['expression'])
                
                shared_vertex_dict = {v:info for v,info in shared_vertex_dict.items() if len(info['bond_vars']) == 2}

                edge_bond_dict = {}
                for btype, blist in edge_bonds.items():
                    for bond in blist:
                        atom1_label = bond["atom1"]["label"]
                        atom2_label = bond["atom2"]["label"]
                        bond_var_name = f"bond_length_{atom1_label}_{atom2_label}_{phase_key}"
                        edge_bond_dict.setdefault(atom2_label, []).append(bond_var_name)
                
                print(f"[INFO] Identified {sum(len(lst) for lst in edge_bond_dict.values())} edge bonds.")
                
                bond_vectors_current = self.analyzer.get_polyhedral_bond_vectors(getattr(self.cpdf, phase).phase)
                problematic_bonds_dict = {}
                for bond_type, blist in bond_vectors_current.items():
                    if 'O-O' in bond_type: continue
                    center_sym = bond_type.split('-')[0]
                    cutoff = self.config.detailed_composition[center_sym]['cutoff']
                    for bond in blist:
                        if not (cutoff[0] <= bond['length'] <= cutoff[1]):
                            bond_key = f"bond_length_{bond['atom1']['label']}_{bond['atom2']['label']}_{phase_key}"
                            problematic_bonds_dict[bond_key] = bond
                
                print(f"[INFO] Detected {len(problematic_bonds_dict)} problematic bonds (outside cutoff).")

                for var_name, data in constrain_dict_bonds.items():
                    try:
                        self.fit.newVar(var_name, tags=['bond_length'])
                        self.fit.constrain(var_name, data['expression'])
                        
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
                        self.fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=sigma)
                        print(f"[{category}] {var_name}: lb={lb:.6f}, ub={ub:.6f}, sigma={sigma}")
                    except Exception as e:
                        print(f"[WARNING] Skipped {var_name} due to error: {e}")

            # --------------------------------------------------------------------
            # ANGLE CONSTRAINTS
            # --------------------------------------------------------------------
            if constrain_angles[0]:
                print("\n[INFO] Processing angle constraints...")
                for var_name, data in constrain_dict_angles.items():
                    try:
                        ang_limit = 1.0
                        self.fit.newVar(var_name, tags=['bond_angle'])
                        self.fit.constrain(var_name, data['expression'])
                        lb = np.radians(data['angle'] - ang_limit)
                        ub = np.radians(data['angle'] + ang_limit)
                        self.fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=constrain_angles[1])
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
                        ang_limit = 2.0
                        self.fit.newVar(var_name, tags=['dihedral_angle'])
                        self.fit.constrain(var_name, data['expression'])
                        lb = np.radians(data['angle'] - ang_limit)
                        ub = np.radians(data['angle'] + ang_limit)
                        self.fit.restrain(var_name, lb=lb, ub=ub, scaled=True, sig=constrain_dihedrals[1])
                        print(f"[INFO] {var_name}: dihedral={data['angle']:.2f}° (±{ang_limit}°)")
                    except Exception as e:
                        print(f"[WARNING] Skipped dihedral {var_name} due to error: {e}")

            # --------------------------------------------------------------------
            # Cleanup: Remove variables that are None and provide detailed debug info
            # --------------------------------------------------------------------
            for name in dir(self.fit):
                if name.startswith(('bond_length_', 'angle_', 'dihedral_')):
                    try:
                        var = getattr(self.fit, name)
                        if var.value is None:
                            # --- Start of new detailed error reporting ---
                            print(f"\n[DEBUG INFO] Found invalid value for '{name}'.")
                            parts = name.split('_')
                            var_type = parts[0]
                            phase_name = parts[-1]
                            atom_labels = parts[1:-1]

                            try:
                                # Get the structure and create an atom lookup map
                                phase_generator = getattr(self.cpdf, phase_name)
                                structure = phase_generator.phase.stru
                                atom_map = {atom.label: atom for atom in structure}

                                print(f"  Atom Coordinates in Phase '{phase_name}':")
                                atom_objects = [atom_map.get(label) for label in atom_labels]
                                
                                if any(atom is None for atom in atom_objects):
                                    print("  Could not find all atoms for this variable.")
                                else:
                                    for atom in atom_objects:
                                        print(f"    - {atom.label}: ({atom.x:.6f}, {atom.y:.6f}, {atom.z:.6f})")

                                    # Attempt a robust recalculation for angles
                                    if var_type == 'angle' and len(atom_objects) == 3:
                                        p0 = np.array([atom_objects[0].x, atom_objects[0].y, atom_objects[0].z])
                                        p1 = np.array([atom_objects[1].x, atom_objects[1].y, atom_objects[1].z])
                                        p2 = np.array([atom_objects[2].x, atom_objects[2].y, atom_objects[2].z])

                                        # Get lattice to convert to Cartesian for accurate angle
                                        phase = phase_generator.phase
                                        lat = phase.lattice
                                        lattice_mat = self.analyzer._lattice_vectors(lat.a.value, lat.b.value, lat.c.value, lat.alpha.value, lat.beta.value, lat.gamma.value)

                                        # Angle is at the central atom (p1)
                                        v1 = lattice_mat.dot(p0 - p1)
                                        v2 = lattice_mat.dot(p2 - p1)
                                        
                                        robust_angle = self.analyzer.calculate_angle(v1, v2)
                                        print(f"  Robust Recalculation: {robust_angle:.4f} degrees.")
                                        if np.isclose(robust_angle, 0.0) or np.isclose(robust_angle, 180.0):
                                            print("  📌 Likely Cause: Atoms are nearly collinear, causing the arccos() calculation to fail.")

                            except Exception as e_debug:
                                print(f"  Could not retrieve full debug info: {e_debug}")
                            # --- End of new detailed error reporting ---

                            self.fit.unconstrain(var)
                            self.fit.delVar(var)
                            print(f"[INFO] Removed '{name}' due to the issue detailed above.\n")
                    except Exception:
                        continue


    def copy_phase_structure(self, cpdf_target, cpdf_source, phase_index=0):
        """
        Copies atomic xyz coordinates between matching atoms of a specific phase.
    
        This function selectively updates the atomic positions (`x`, `y`, `z`) in a
        target structure (`cpdf_target`) with values from a source structure
        (`cpdf_source`). It operates on a single phase, identified by its index.
    
        The core of the function is its atom-matching mechanism, which relies on
        assigning unique labels (e.g., 'O1', 'O2', 'Si1') to atoms in both
        structures. It then iterates through the target atoms, and for each one
        that has a matching label in the source, it copies the coordinates. This
        allows for a precise transfer of positional information without affecting
        other parameters like lattice constants or thermal parameters.
    
        Args:
            cpdf_target (PDFContribution):
                The `PDFContribution` object whose structure will be updated. This
                object is modified in-place.
            cpdf_source (PDFContribution):
                The `PDFContribution` object that contains the source structure
                from which coordinates will be read.
            phase_index (int, optional):
                The numerical index of the phase to copy within the
                `PDFContribution` (e.g., 0 for "Phase0", 1 for "Phase1").
                Defaults to 0.
    
        Returns:
            None. The function modifies the `cpdf_target` object directly.
    
    
        Notes:
            - The function's success hinges on the `assignUniqueLabels()` method
              generating corresponding labels for atoms in both structures. This
              is generally reliable when the source and target structures have the
              same composition and atom ordering.
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
    def compare_fits(self, fitA, fitB, tol=1e-8):
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
            
    def rebuild_contribution(self, old_cpdf, r_obs, g_obs, name='cpdf_rebuilt'):
        """
        Rebuilds a PDFContribution using existing structures with new data.
    
        This function creates a new `PDFContribution` object. Instead of loading
        structures from their original source files (e.g., CIFs), it extracts the
        `Structure` objects directly from an existing `PDFContribution`. This is
        a crucial operation in sequential analysis, as it preserves the exact
        state of a structure (including atomic positions, lattice, etc.) after a
        refinement.
    
    
        Args:
            old_cpdf (PDFContribution):
                The source `PDFContribution` containing the refined structures
                that will be copied and preserved.
            r_obs (numpy.ndarray):
                The new observed r-grid (independent variable) for the rebuilt
                contribution.
            g_obs (numpy.ndarray):
                The new observed G(r) data (dependent variable) for the rebuilt
                contribution.
            name (str, optional):
                A string identifier for the new `PDFContribution`. Defaults to
                'cpdf_rebuilt'.
    
        Returns:
            PDFContribution:
                A new, fully configured `PDFContribution` instance. This object
                contains independent copies of the structures from `old_cpdf` but
                is associated with the new observed data and calculation settings
                derived from the parent class instance.
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

        # Use self.config to get myrange and myrstep
        cpdf_new.setCalculationRange(xmin=self.config.myrange[0], xmax=self.config.myrange[1], dx=self.config.myrstep)
        print(f"[DEBUG] Set calculation range: r ∈ [{self.config.myrange[0]}, {self.config.myrange[1]}], Δr = {self.config.myrstep}")

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
                # Keep evaluator type consistent
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

            # Use self.ncpu and self.pool from the PDFRefinement object
            new_gen.parallel(ncpu=self.ncpu, mapfunc=self.pool.map)
            print(f"[DEBUG]  - Enabled parallel with ncpu={self.ncpu}")

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


    def initialize_from_special_structure(self):
        """Initializes model coordinates from an external structure file.
    
        This function acts as a specialized setup routine for a refinement
        workflow. It checks the main configuration for a 'special_structure'
        entry. If found, it reads the specified structure file (e.g., a CIF
        from a previous refinement or a theoretical model) and uses it to
        overwrite the initial atomic positions (`x`, `y`, `z`) of the
        corresponding phase in the main refinement model (`self.cpdf`).
    
        This is particularly useful for starting a new analysis from a known
        good starting point, bypassing the default structure loaded from the
        standard input files.
    
        Returns:
            None. The function modifies the instance's `self.cpdf` object
            in-place.
        """
        if hasattr(self.config, 'special_structure'):
            print("\n---------------------------------------------------------")
            print("           INITIALIZING FROM SPECIAL STRUCTURE           ")
            print("---------------------------------------------------------")
            
            try:
                # 1. Get details from the configuration
                spec_config = self.config.special_structure
                file_path = spec_config['file_path']
                phase_idx = spec_config.get('phase_index_to_update', 0)
                phase_name = f"Phase{phase_idx}"
    
                print(f"Loading special structure from: {file_path}")
                print(f"Updating coordinates for phase: {phase_name}")
    
                # 2. Load the source structure directly from the CIF file
                source_stru = loadStructure(file_path)
    
                # 3. Get the target structure from the main cpdf object
                target_phase_generator = getattr(self.cpdf, phase_name)
                target_stru = target_phase_generator.stru
    
                # 4. Perform the coordinate copy using the helper method
                self._copy_atom_positions_from_structure(target_stru, source_stru)
                print("---------------------------------------------------------\n")
    
            except FileNotFoundError:
                print(f"[ERROR] Special structure file not found: {file_path}")
            except (KeyError, AttributeError) as e:
                print(f"[ERROR] Could not find phase '{phase_name}' in the model or config is malformed: {e}")
            except Exception as e:
                print(f"[ERROR] An unexpected error occurred while loading the special structure: {e}")
    
    def _copy_atom_positions_from_structure(self, target_stru, source_stru):
        """Copies atomic coordinates from a source to a target structure.

        This private helper method performs an in-place update of the atomic
        `x`, `y`, `z` coordinates in the `target_stru`. It works by matching
        atoms between the source and target structures based on unique,
        automatically generated labels (e.g., 'O1', 'Zr1'). For every atom in
        the target that has a counterpart in the source, its coordinates are
        overwritten.
        
        Args:
            target_stru (diffpy.structure.Structure):
                The structure object whose atomic coordinates will be modified
                in-place.
            source_stru (diffpy.structure.Structure):
                The structure object from which the new coordinates will be read.
        
        Returns:
            None.
        """
        # Create a lookup map from the source structure's atom labels to atoms
        source_stru.assignUniqueLabels()
        src_map = {atom.label: atom for atom in source_stru}
    
        # Iterate through the target structure and update coordinates
        n_copied = 0
        target_stru.assignUniqueLabels()
        for atom in target_stru:
            if atom.label in src_map:
                old_coords = (atom.x, atom.y, atom.z)
                src_atom = src_map[atom.label]
                new_coords = (src_atom.x, src_atom.y, src_atom.z)
                
                # Update coordinates in the target structure
                atom.x, atom.y, atom.z = new_coords
                
                print(f"Atom {atom.label}: ({old_coords[0]:.5f}, {old_coords[1]:.5f}, {old_coords[2]:.5f}) → "
                      f"({new_coords[0]:.5f}, {new_coords[1]:.5f}, {new_coords[2]:.5f})")
                n_copied += 1
        print(f"\nCopied xyz coordinates for {n_copied} matching atoms.")

            
    #=============================================================================
    # # Simulate PDFs
    # # =============================================================================
    def simulate_pdf_workflow(self, main_config, sim_config):
        """Runs a complete, self-contained PDF simulation workflow.
        
            This function orchestrates a full simulation process, distinct from a
            refinement. Its purpose is to take a structural model (from a CIF file),
            manually set all relevant physical parameters to specific, pre-defined
            values, calculate the resulting theoretical PDF, and save all outputs.
            This is ideal for generating theoretical patterns, testing the effect of
            specific parameters, or creating a baseline model from known values.
        
            The workflow is data-driven, using a main configuration for global
            settings and a dedicated simulation dictionary for the specific inputs,
            parameters, and output locations.
        
            Args:
                main_config (RefinementConfig):
                    The main project configuration object, providing global settings
                    like the path to the raw diffraction data, Qmax, r-range, etc.
                sim_config (dict):
                    A dictionary containing all parameters specific to this
                    simulation run. It must have the following structure:
                    {
                        "output_path": str,  # Directory to save results.
                        "powder_data_file": str,  # Raw data file (e.g., .xy) to get r-grid from.
                        "cif_directory": str, # Path to the CIF file's directory.
                        "ciffile": str,      # Filename of the CIF for the structure model.
                        "fitting_range": tuple[float, float], # (rmin, rmax) for final evaluation.
                        "csv_filename": str, # Name for the output data file.
                        "optimized_params": {
                            "Phase0": {      # A key for each phase (e.g., "Phase0")
                                "s": float,      # Scale factor
                                "psize": float,  # Nanoparticle size
                                "delta2": float  # Step-function broadening
                            },
                            # ... other phases ...
                        },
                        "default_Uiso": {
                            "Ti": float,     # Isotropic ADP for Titanium atoms
                            "O": float,      # Isotropic ADP for Oxygen atoms
                            # ... other elements ...
                        }
                    }
        
            Returns:
                PDFContribution:
                    The final, fully configured `PDFContribution` object containing the
                    structure and all injected parameters used for the simulation.
        
            Side Effects:
                - Creates the output directory specified in `sim_config` if it
                  does not exist.
                - Writes a CSV file containing the observed (from data), calculated
                  (simulated), and difference profiles to the output directory.
                - Generates and saves summary plots (e.g., bond lengths, angles)
                  for each phase to the output directory.
        
            Example:
                >>> # Assume 'workflow' is an instance of the parent class.
                >>> main_config = load_main_config('project.yml')
                >>> sim_settings = {
                ...     "output_path": "./simulation_results/",
                ...     "powder_data_file": "raw_data.xy",
                ...     "cif_directory": "./structures/",
                ...     "ciffile": "rutile_model.cif",
                ...     "fitting_range": (1.0, 20.0),
                ...     "csv_filename": "simulated_rutile.csv",
                ...     "optimized_params": {
                ...         "Phase0": {"s": 0.95, "psize": 150.0, "delta2": 2.1}
                ...     },
                ...     "default_Uiso": {"Ti": 0.005, "O": 0.007}
                ... }
                >>>
                >>> final_model = workflow.simulate_pdf_workflow(main_config, sim_settings)
                >>> # This will create the './simulation_results' directory and populate
                >>> # it with 'simulated_rutile.csv' and various structural plots.
            """
        print("\n==========================================================")
        print("           STARTING PDF SIMULATION WORKFLOW           ")
        print("==========================================================")
        
        # --- Setup ---
        output_path = sim_config['output_path']
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            print(f"Created simulation output directory: {output_path}")

        # --- PDF Generation (Corrected Call) ---
        # The call now matches the flexible generatePDF method signature.
        r0_sim, g0_sim, cfg_sim = self.pdf_manager.generatePDF(
            data_directory=main_config.xrd_directory,
            data_filename=sim_config['powder_data_file'],
            composition=main_config.composition,
            qmax=main_config.qmax,
            myrange=main_config.myrange,
            myrstep=main_config.myrstep,
            pdfgetx_config=main_config.pdfgetx_config
        )

        # --- Model Building ---
        # Temporarily set the cif_directory for this specific build.
        original_cif_dir = self.config.cif_directory
        self.config.cif_directory = sim_config['cif_directory']
        cpdf_sim = self.pdf_manager.build_contribution(
            r0_sim, g0_sim, cfg_sim, sim_config['ciffile'], main_config.myrange
        )
        self.config.cif_directory = original_cif_dir # Restore original path

        # --- Parameter Injection ---
        # 1. Register spherical envelope function and build the equation
        phase_equation = ""
        for phase in cpdf_sim._generators:
            cpdf_sim.registerFunction(
                sphericalCF,
                name=f"sphere_{phase}",
                argnames=["r", f"psize_{phase}"]
            )
            phase_equation += f"s_{phase}*{phase}*sphere_{phase} + "
        cpdf_sim.setEquation(phase_equation.rstrip(" + "))

        # 2. Inject optimized fit parameters from the sim_config dictionary
        optimized_params = sim_config['optimized_params']
        for phase, params in optimized_params.items():
            getattr(cpdf_sim, f"s_{phase}").value = params['s']
            setattr(cpdf_sim, f"psize_{phase}", params['psize'])
            getattr(cpdf_sim, str(phase)).delta2.value = params['delta2']
        
        # 3. Set isotropic atomic displacement parameters (Uiso) by element
        default_Uiso = sim_config['default_Uiso']
        for phase in cpdf_sim._generators:
            struct = getattr(cpdf_sim, str(phase)).phase
            for atom in struct.getScatterers():
                if atom.element in default_Uiso:
                    atom.Uiso.value = default_Uiso[atom.element]

        # --- Evaluation and Results ---
        # 1. Evaluate fit over the specified range and save the data
        self.results_manager.evaluate_and_plot(
            cpdf_sim,
            fitting_range=sim_config['fitting_range'],
            csv_filename=os.path.join(output_path, sim_config['csv_filename'])
        )

        # 2. Generate and save the summary plots (bond lengths, angles, etc.)
        for phase_name in cpdf_sim._generators:
            phase_obj = getattr(cpdf_sim, str(phase_name)).phase
            plot_prefix = os.path.join(output_path, f"{phase_name}_")
            self.results_manager.visualize_fit_summary(
                cpdf_sim,
                phase_obj,
                plot_prefix
            )
        
        print("\nSimulation workflow finished successfully.")
        return cpdf_sim
    

class PDFWorkflowManager(PDFRefinement):
    """
    An alternative controller to manage a sequence of refinements on a list
    of datasets. It includes logging and a checkpoint/resume system.
    """
    def __init__(self, config, pdf_manager, results_manager, helper, analyzer, ncpu, pool):
        """Initializes the sequential refinement workflow."""
        super().__init__(config, pdf_manager, results_manager, helper, analyzer, ncpu, pool)
       
        # Dynamically construct the checkpoint directory path
        self.checkpoint_dir = os.path.join(self.config.fit_directory, self.config.project_name, 'checkpoints')
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Setup logging
        log_file = os.path.join(self.config.fit_directory, self.config.project_name, self.config.log_file)
        self.results_manager.setup_logging(log_file)

    def _get_checkpoint_paths(self, dataset_id):
        """Generate standardized paths for checkpoint files."""
        base_name = os.path.splitext(dataset_id)[0]
        params_path = os.path.join(self.checkpoint_dir, f"{base_name}_params.json")
        cif_path = os.path.join(self.checkpoint_dir, f"Phase0_checkpoint_{base_name}.cif")
        return params_path, cif_path

    def save_checkpoint(self, dataset_id, step_index):
                """Saves the current state, including the last completed step."""
                params_path, _ = self._get_checkpoint_paths(dataset_id)
                self.log(f"Saving checkpoint for {dataset_id} after step {step_index}...")
                try:
                    # Extract parameters from the FitRecipe
                    params_to_save = {}
                    for name in self.fit.names:
                        params_to_save[name] = getattr(self.fit, name).value
                    
                    # Create a checkpoint dictionary that includes the progress
                    checkpoint_data = {
                        'last_completed_step': step_index,
                        'parameters': params_to_save
                    }
                    
                    # Save the combined data to the JSON file
                    with open(params_path, 'w') as f:
                        json.dump(checkpoint_data, f, indent=4)
                    
                    # CORRECTED: Save the CIF to the central checkpoint directory
                    base_name = os.path.splitext(dataset_id)[0]
                    cif_file_tag = f"checkpoint_{base_name}"
                    self.results_manager.export_cifs(cif_file_tag, self.cpdf, output_dir=self.checkpoint_dir)
                    
                    self.log(f"  - Structure saved.")
                    self.log(f"  - Parameters and progress saved to {params_path}")
                except Exception as e:
                    self.log(f"Error saving checkpoint for {dataset_id}: {e}", level='error')


    def load_checkpoint(self, dataset_id):
            """Loads checkpoint data (progress and parameters) and returns them."""
            params_path, cif_path = self._get_checkpoint_paths(dataset_id)
        
            if os.path.exists(params_path) and os.path.exists(cif_path):
                self.log(f"Found checkpoint for {dataset_id}. Loading...")
                try:
                    # Load the entire checkpoint data object from JSON
                    with open(params_path, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    # Set the special structure path in config to the checkpoint CIF
                    self.config.special_structure = {
                        'file_path': cif_path,
                        'phase_index_to_update': 0
                    }
                    self.log(f"  - Checkpoint loaded successfully. Last completed step was {checkpoint_data.get('last_completed_step', 'N/A')}.")
                    return checkpoint_data # Return the full dictionary
                except Exception as e:
                    self.log(f"Error loading checkpoint for {dataset_id}: {e}", level='error')
                    return None
            return None

    def run_sequential_workflow(self):
                """
                Executes the entire automated refinement workflow.
            
                This method serves as the main entry point for the analysis. It iterates
                through a list of datasets provided in the configuration and applies a
                multi-step refinement, as defined in the 'refinement_plan'.
            
                The key functionalities include:
                - Processing a sequence of one or more datasets.
                - A checkpoint system that saves the state after each refinement step,
                    allowing the process to be resumed if interrupted.
                - For sequential datasets, the refined model from the previous dataset is
                    used as the starting point for the next, ensuring continuity.
                - For the very first dataset, it can optionally start from a user-provided
                    'special_structure' CIF file.
                """
                if not hasattr(self.config, 'refinement_plan'):
                    self.log("Error: 'refinement_plan' not found in the configuration.", level='error')
                    return
            
                refinement_plan = self.config.refinement_plan
                self.log("======= STARTING REFINEMENT WORKFLOW =======")
                
                # Keep track of the last successful fit AND its corresponding cpdf
                last_successful_fit = None
                last_successful_cpdf = None
            
                for idx, dataset in enumerate(self.config.dataset_list):
                    dataset_id = os.path.basename(dataset)
                    self.log(f"\n===== PROCESSING DATASET {idx + 1}/{len(self.config.dataset_list)}: {dataset_id} =====")
            
                    self.config.new_output_directory(subdir_name=os.path.splitext(dataset_id)[0])
                    r, g, cfg = self.pdf_manager.generatePDF(
                        data_directory=self.config.xrd_directory, data_filename=dataset,
                        composition=self.config.composition, qmax=self.config.qmax,
                        myrange=self.config.myrange, myrstep=self.config.myrstep,
                        pdfgetx_config=self.config.pdfgetx_config
                    )
                    
                    start_step = 0
                    checkpoint_data = self.load_checkpoint(dataset_id)
        
                    if checkpoint_data:
                        # SCENARIO A: RESUME from a checkpoint.
                        self.log("Resuming from checkpoint.")
                        # Build a temporary cpdf just to load the structure into
                        self.cpdf = self.pdf_manager.build_contribution(r, g, cfg, self.config.ciffile, self.config.myrange)
                        self.fit = self.build_initial_recipe()
                        
                        loaded_params = checkpoint_data.get('parameters', {})
                        self.log("Applying loaded parameters to the model...")
                        for name, value in loaded_params.items():
                            if name in self.fit.names:
                                getattr(self.fit, name).value = value
                        
                        self.initialize_from_special_structure()
                        last_step = checkpoint_data.get('last_completed_step', -1)
                        start_step = last_step + 1
        
                    elif idx > 0 and last_successful_fit and last_successful_cpdf:
                        # SCENARIO B: CONTINUE from previous dataset's results.
                        self.log("Initializing from previous dataset's results.")
                        
                        # **FIX:** Rebuild the contribution using the PREVIOUS refined structure
                        self.cpdf = self.rebuild_contribution(
                            old_cpdf=last_successful_cpdf,
                            r_obs=r,
                            g_obs=g,
                            name=f'cpdf_{idx}'
                        )
                        
                        new_fit = self.build_initial_recipe()
                        self.update_recipe_from_initial(last_successful_fit, new_fit, self.cpdf, recalculate_bond_vectors=True)
        
                    else:
                        # SCENARIO C: START new refinement from scratch.
                        self.log("First dataset: Building initial model from scratch.")
                        self.cpdf = self.pdf_manager.build_contribution(r, g, cfg, self.config.ciffile, self.config.myrange)
                        self.initialize_from_special_structure()
                        self.fit = self.build_initial_recipe()
        
                    # --- REFINEMENT LOOP ---
                    for i in range(start_step, len(refinement_plan)):
                        step_params = refinement_plan[i]
                        self.log(f"\n--- Running Step {i}: {step_params.get('description', 'No description')} ---")
            
                        if 'space_group' in step_params:
                            self.modify_recipe_spacegroup(step_params['space_group'])
            
                        if 'constraints' in step_params:
                            self.apply_rigid_body_constraints(**step_params['constraints'])
            
                        self.run_refinement_step(
                            i,
                            step_params['fitting_range'],
                            self.config.myrstep,
                            step_params['fitting_order'],
                            'resv'
                        )
                        self.save_checkpoint(dataset_id, i)
            
                    self.results_manager.finalize_results(self.cpdf, self.fit)
                    
                    # Update both trackers for the next iteration
                    last_successful_fit = self.fit
                    last_successful_cpdf = self.cpdf
                    
                    self.log(f"===== COMPLETED DATASET: {dataset_id} =====")
            
                self.log("\n======= REFINEMENT WORKFLOW FINISHED =======")
