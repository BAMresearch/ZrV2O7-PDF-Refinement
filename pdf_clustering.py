"""
================================================================================
  Unsupervised Learning and Clustering Analysis of PDF Sequential Data
================================================================================

Author: Tomasz Stawski (tomasz.stawski@bam.de, tomasz.stawski@gmail.com)
Testing, Implementation and Analyses: Aiste Miliute (aiste.miliute@bam.de)
Version: 1.3.0
License: MIT License

DESCRIPTION:
This script performs an unsupervised statistical analysis on a sequential 
series of Pair Distribution Function (PDF) curves. It applies multiple 
clustering and dimensionality reduction techniques to identify structural 
phase transitions and correlate continuous structural evolution.

Analyses include Spectral Clustering, Non-negative Matrix Factorisation (NMF), 
Correlation Analysis, Principal Component Analysis (PCA), and Hierarchical 
Clustering.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Math & Clustering Libraries
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA

# PDF generation imports
# Note: Ensure diffpy.pdfgetx is installed in your environment
try:
    from diffpy.pdfgetx import PDFConfig, PDFGetter
    DIFFPY_AVAILABLE = True
except ImportError:
    print("WARNING: diffpy.pdfgetx not found. PDF generation will fail.")
    DIFFPY_AVAILABLE = False

# =============================================================================
# 1. USER PARAMETERS
# =============================================================================
data_dir    = 'dataXRD'             # Directory containing raw .dat scattering files
data_dir_gr = 'dataXRDandGofRs'     # Directory containing precalculated .gr files (Fallback)
output_dir  = 'ClusteringAnalysis'  # Output directory for saved plots and matrices
composition = 'O7 V2 Zr1'           # Specified target system stoichiometry
qmin, qmax  = 0.0, 22.0             # Q-range limits applied during PDF generation
r_range     = (0.05, 80)            # Discrete r-range bounds (rmin, rmax) for the calculation
myrstep     = 0.05                  # Incremental grid spacing (Δr)
n_clusters  = 3                     # Targeted number of clusters for initial partitioning models

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def load_gr_data(filepath):
    """Robustly reads precomputed .gr files bypassing complex headers."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
    return np.array(data)

def generatePDF(filepath, composition, qmin=0.0, qmax=22.0, rmin=1.0, rmax=100.0, rstep=0.01):
    """
    Generates the Pair Distribution Function G(r) from raw diffraction data using diffpy.pdfgetx.

    Args:
        filepath (str): Path to the raw diffraction data file.
        composition (str): Chemical stoichiometry of the sample (e.g., 'O7 V2 Zr1').
        qmin (float, optional): Minimum momentum transfer Q threshold. Defaults to 0.0.
        qmax (float, optional): Maximum momentum transfer Q threshold. Defaults to 22.0.
        rmin (float, optional): Minimum radial distance for the calculated PDF. Defaults to 1.0.
        rmax (float, optional): Maximum radial distance for the calculated PDF. Defaults to 100.0.
        rstep (float, optional): Incremental radial step size. Defaults to 0.01.

    Returns:
        tuple: A tuple containing two 1D numpy arrays:
            - r (ndarray): The discrete radial grid values.
            - g (ndarray): The corresponding experimental PDF intensity values G(r).
            
    Raises:
        ImportError: If the 'diffpy.pdfgetx' library is not installed in the environment.
    """
    if not DIFFPY_AVAILABLE:
        raise ImportError("Diffpy is not installed.")
        
    cfg = PDFConfig(mode='xray',
                    composition=composition,
                    dataformat='QA',
                    rpoly=1.3,
                    rstep=rstep,
                    rmin=rmin,
                    rmax=rmax)
    cfg.qmin = qmin
    cfg.qmax = cfg.qmaxinst = qmax
    pg = PDFGetter(config=cfg)
    # Return r, g
    return pg(filename=filepath)

def spectral_cluster_curves(curves, n_clusters=2, sigma=None):
    """
    Performs spectral clustering on a series of structural curves.
    
    Spectral clustering works by treating each distribution curve as a node in an 
    undirected graph. It connects these nodes based on their pairwise Euclidean 
    dissimilarity, weights them via a Gaussian kernel (sigma), and groups them 
    by solving the eigenvalue problem of the resulting graph Laplacian matrix.
    This effectively isolates distinct structural 'regimes' or phases even when 
    the boundaries between them are non-linear or diffuse.

    Args:
        curves (ndarray): A 2D numpy array of shape (n_samples, n_features) containing 
                          the sequential structural data curves to cluster.
        n_clusters (int, optional): The targeted quantity of discrete phase clusters to extract. Defaults to 2.
        sigma (float, optional): The Gaussian kernel width parameter. If None, the median 
                                 Euclidean distance across the dataset is automatically computed. Defaults to None.

    Returns:
        ndarray: A 1D array of integers (shape: (n_samples,)) representing the discrete 
                 cluster label assigned to each corresponding input curve.
    """
    # 1. Similarity Matrix
    D = squareform(pdist(curves, metric='euclidean'))
    sigma = np.median(D) if sigma is None else sigma
    W = np.exp(-D**2 / (2*sigma**2))
    
    # 2. Laplacian
    row_sum = W.sum(axis=1)
    row_sum[row_sum == 0] = 1e-10 # Avoid division by zero
    D_inv_sqrt = np.diag(1/np.sqrt(row_sum))
    L = np.eye(len(curves)) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # 3. Eigen decomposition
    eigvals, eigvecs = eigh(L)
    U = eigvecs[:, 1:n_clusters]
    
    # 4. Normalise and Cluster
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms==0] = 1e-10
    U_norm = U / norms
    
    return KMeans(n_clusters=n_clusters, random_state=0).fit_predict(U_norm)

def sort_linkage_by_temperature(Z, temperatures):
    """
    Reorders left/right children in each row of a scipy linkage matrix so
    that within every branch the lower-mean-temperature cluster appears on
    the left and the higher-mean-temperature cluster appears on the right.

    scipy's dendrogram renders Z[k, 0] on the left and Z[k, 1] on the right,
    so swapping these two indices directly controls the leaf ordering without
    altering the underlying clustering geometry or Ward distances.

    Args:
        Z           (ndarray): Linkage matrix returned by scipy.cluster.hierarchy.linkage,
                               shape (n-1, 4). A modified copy is returned.
        temperatures (ndarray): 1-D array of temperature values (length n), already
                                sorted to match the original row order of the feature
                                matrix passed to linkage().

    Returns:
        ndarray: A copy of Z with left/right children swapped where necessary.
    """
    n          = len(temperatures)
    Z_new      = Z.copy().astype(float)
    mean_temps = np.empty(2 * n - 1)
    mean_temps[:n] = temperatures   # leaf nodes carry their own temperature

    for k in range(len(Z_new)):
        c1 = int(Z_new[k, 0])
        c2 = int(Z_new[k, 1])
        t1 = mean_temps[c1]
        t2 = mean_temps[c2]

        # Swap so that the colder cluster is always drawn on the left
        if t1 > t2:
            Z_new[k, 0], Z_new[k, 1] = c2, c1
            t1, t2 = t2, t1

        # Mean temperature of the newly formed internal node
        mean_temps[n + k] = (t1 + t2) / 2

    return Z_new

# =============================================================================
# 3. DATA LOADING WORKFLOW
# =============================================================================
print("--- Loading and Generating PDFs ---")
pattern   = os.path.join(data_dir, 'PDF_ZrV2O7_061_*C_avg_*.dat')
filepaths = sorted(glob.glob(pattern))

temperatures = []
all_curves   = []
r_ref        = None

if not filepaths:
    print(f"No files found matching: {pattern}")
    print("Please check your 'data_dir' and file naming pattern.")
    # In a real run, we would exit here, but for safety in copy-paste:
    # exit()

for fp in filepaths:
    # Extract temperature from filename (expects format like "..._300C_avg...")
    m = re.search(r'_(\d+)C_avg', os.path.basename(fp))
    if not m: continue
    T = int(m.group(1))
    
    try:
        if DIFFPY_AVAILABLE:
            r, g = generatePDF(fp, composition,
                               qmin=qmin, qmax=qmax,
                               rmin=r_range[0], rmax=r_range[1], rstep=myrstep)
        else:
            # FALLBACK: Load pre-calculated .gr file if diffpy.pdfgetx is missing
            gr_fname = os.path.basename(fp).replace('.dat', '.gr')
            gr_path  = os.path.join(data_dir_gr, gr_fname)
            
            if not os.path.exists(gr_path):
                raise FileNotFoundError(f"Fallback .gr file not found: {gr_path}")
            
            data = load_gr_data(gr_path)
            if len(data) == 0:
                raise ValueError(f"No valid numerical data extracted from {gr_path}")
            raw_r = data[:, 0]
            raw_g = data[:, 1]
            
            # Interpolate to precisely match the target physical grid bounds
            # np.arange includes up to rmax + myrstep/2 to ensure inclusive upper bound
            r = np.arange(r_range[0], r_range[1] + myrstep/2, myrstep)
            f_interp = interp1d(raw_r, raw_g, kind='linear', bounds_error=False, fill_value=0.0)
            g = f_interp(r)
            
        temperatures.append(T)
        all_curves.append(g)
        if r_ref is None: r_ref = r
        print(f"Loaded: {os.path.basename(fp)} -> {T}°C {'(diffpy realtime)' if DIFFPY_AVAILABLE else '(fallback precalculated)'}")
    except Exception as e:
        print(f"Failed to process {fp}: {e}")

if not all_curves:
    print("No data loaded. Exiting.")
    exit()

# Stack into matrix: (n_samples, n_r_points)
pdf_curves  = np.vstack(all_curves)
temperatures = np.array(temperatures)

# SORT BY TEMPERATURE ---
# Glob loads files alphabetically (100, 1000, 200). We must sort numerically.
sort_idx = np.argsort(temperatures)
temperatures = temperatures[sort_idx]
pdf_curves   = pdf_curves[sort_idx]
print(f"Data Sorted. T Range: {temperatures.min()} - {temperatures.max()}°C")

# =============================================================================
# NORMALISATION: HIGH-R TAIL BASELINE (r > 35 Å)
# =============================================================================
# Purpose: Normalise datasets derived from the intensity of the long-range 
# structural tail, operating on the premise that high-r structural signals 
# maintain scale invariance.

if r_ref is None:
    print("Error: No r-axis data found. Cannot filter by range.")
    exit()

# 1. Define the Cutoff
norm_r_min = 35.0
mask_high_r = r_ref > norm_r_min

if np.sum(mask_high_r) == 0:
    raise ValueError(f"No data points found for r > {norm_r_min} Å. "
                     "High-r normalisation is mandatory to preserve correct scaling.")

# 2. Calculate Scale Factor from the High-r Segment ONLY
# We extract just the tail (r > 35) and calculate its "energy" (L2 norm)
high_r_segment = pdf_curves[:, mask_high_r]
scale_factors = np.linalg.norm(high_r_segment, axis=1)

# 3. Apply Scaling to the FULL Curve
# We divide the entire G(r) by the scale factor derived from the tail
# Reshape (N,) -> (N,1) for broadcasting
scale_factors[scale_factors == 0] = 1.0 # Safety check
pdf_curves = pdf_curves / scale_factors[:, np.newaxis]

print(f"Data normalised referencing the r > {norm_r_min} Å region.")
# =============================================================================

# =============================================================================
# OPTIONAL: Visualise Normalisation
# =============================================================================
# Employs the foundational reference vector (r_ref) for spatial plotting.
if r_ref is not None:
    plt.figure(figsize=(10, 5))
    
    # Visualise a representative subset to prevent graphical clutter
    subset_indices = np.linspace(0, len(temperatures)-1, 5, dtype=int)
    
    for i in subset_indices:
        plt.plot(r_ref, pdf_curves[i], label=f'{temperatures[i]}°C')
        
    plt.xlabel("r (Å)")
    plt.ylabel("Normalised Intensity")
    plt.title(f"Normalised PDFs (Subset of {len(subset_indices)} distributions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '00_Normalised_PDFs.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '00_Normalised_PDFs.svg'))
    plt.show()

# =============================================================================
# 4. SPECTRAL CLUSTERING & NMF ANALYSIS
# =============================================================================
# This section contrasts two opposing analytical philosophies:
# Discrete clustering (Spectral) vs Continuous compositional evolution (NMF).

print("\n--- Running Spectral Clustering & NMF ---")

# A) Spectral Clustering (Discrete Phase Grouping)
# We partition the sequential datasets into hard, discrete clusters. This helps
# rapidly identify strict boundaries for potential structural phase transitions
# occurring over the temperature series.
labels_spec = spectral_cluster_curves(pdf_curves, n_clusters=n_clusters)

# B) NMF (Continuous Non-negative Matrix Factorisation)
# Unlike Spectral Clustering, NMF does not force a dataset into a single rigid category.
# Instead, it assumes the total observed signal is a continuous linear combination of 
# pure "basis" structural components. By tracking the mathematical weights of these 
# components across the sequence, we can map gradual structural phase conversions 
# (e.g. transitioning from Phase A into Phase B via intermediate mixed states).
# Note: NMF mathematically requires all input signals to be non-negative.
minval = pdf_curves.min()
pdf_nn = pdf_curves - minval if minval < 0 else pdf_curves.copy()

nmf = NMF(n_components=n_clusters, init='nndsvda', random_state=0, max_iter=1000)
W = nmf.fit_transform(pdf_nn)   # Weights (n_samples, n_components)
H = nmf.components_             # Basis Vectors (n_components, n_r)

# Reporting DataFrames and saving Data
df_labels = pd.DataFrame({'Temperature (°C)': temperatures, 'SpectralCluster': labels_spec})
df_nmf = pd.DataFrame(W, columns=[f'NMF_comp{i+1}' for i in range(n_clusters)])
df_nmf['Temperature (°C)'] = temperatures

df_labels.to_csv(os.path.join(output_dir, '01_Spectral_Clusters.csv'), index=False)
df_nmf.to_csv(os.path.join(output_dir, '02_NMF_Weights.csv'), index=False)

# Plot 1: Spectral Clustering
plt.figure(figsize=(7,4))
plt.scatter(df_labels['Temperature (°C)'], df_labels['SpectralCluster'],
            c=df_labels['SpectralCluster'], cmap='tab10', s=60, edgecolors='k')
plt.xlabel('Temperature (°C)')
plt.ylabel('Spectral Cluster Label')
plt.title('Method 1: Spectral Clustering (Discrete Zones)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_Spectral_Clustering.png'), dpi=300)
plt.savefig(os.path.join(output_dir, '01_Spectral_Clustering.svg'))
plt.show()

# Plot 2: NMF Weights
plt.figure(figsize=(8,5))
for i in range(n_clusters):
    plt.plot(df_nmf['Temperature (°C)'], df_nmf[f'NMF_comp{i+1}'],
             '-o', label=f'Component {i+1}')
plt.xlabel('Temperature (°C)')
plt.ylabel('Weight (Composition)')
plt.title('Method 2: NMF Component Weights (Continuous Evolution)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_NMF_Weights.png'), dpi=300)
plt.savefig(os.path.join(output_dir, '02_NMF_Weights.svg'))
plt.show()

# =============================================================================
# 5. NEW ANALYSIS: Correlation Matrix (Topology)
# =============================================================================
# Constructing a 2D topographical dissimilarity map across all states simultaneously.

print("\n--- Running Correlation Analysis ---")

# Calculate Correlation Distance Matrix based on the Pearson Correlation coefficient.
# Unlike dimensionality reduction (PCA/NMF), a correlation map rigidly retains the full 
# native dimensionality of the input data. We cross-correlate every single G(r) 
# curve statically against every other curve in the series. 
# 
# Visually analysing this matrix allows direct identification of structural continuity.
#   - Large bright 'blocks' along the diagonal represent highly stable structural regimes.
#   - Sharp transitions to dark boundaries indicate sudden, discontinuous phase changes 
#     or rapid internal unit-cell degradation.
D_corr = squareform(pdist(pdf_curves, metric='correlation'))
df_corr = pd.DataFrame(D_corr, index=temperatures, columns=temperatures)
df_corr.to_csv(os.path.join(output_dir, '03_Correlation_Matrix.csv'))

plt.figure(figsize=(8, 7))
sns.heatmap(D_corr, 
            xticklabels=temperatures, 
            yticklabels=temperatures, 
            cmap='viridis_r', 
            square=True)
plt.title("Method 3: Dissimilarity Matrix\n(Bright = Similar, Dark = Different)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_Correlation_Heatmap.png'), dpi=300)
plt.savefig(os.path.join(output_dir, '03_Correlation_Heatmap.svg'))
plt.show()

# =============================================================================
# 6. NEW ANALYSIS: Principal Component Analysis (PCA) - 3 Components
# =============================================================================
# PCA mathematically isolates the vectors of maximal variance across the entire 
# dataset sequence, allowing us to disentangle distinct, uncorrelated mechanisms 
# driving the physical structural evolution.

print("\n--- Running PCA Analysis (3 Components) ---")

# We extract orthogonal basis vectors capturing the absolute highest variance 
# across the sequence. Physically, these represent the primary driving factors 
# acting upon the material structure:
#   - PC1 (Highest Variance): Typically maps the dominant average structural motif 
#     or the single most robust macro-transition in the dataset.
#   - PC2 (Second Highest): Often captures secondary physical effects, such as 
#     thermal expansion vectors or specific unit-cell distortion modes.
#   - PC3: Captures subtle tertiary structural nuances.
pca = PCA(n_components=3)
pca_weights = pca.fit_transform(pdf_curves) 
explained_var = pca.explained_variance_ratio_ * 100

# Save continuous PCA trajectory data
df_pca = pd.DataFrame(pca_weights, columns=['PC1', 'PC2', 'PC3'])
df_pca['Temperature (°C)'] = temperatures
df_pca.to_csv(os.path.join(output_dir, '04_PCA_Trajectories.csv'), index=False)

# Print explained variance to console
print(f"Explained Variance: PC1={explained_var[0]:.1f}%, PC2={explained_var[1]:.1f}%, PC3={explained_var[2]:.1f}%")

# --- Visualization ---
fig = plt.figure(figsize=(14, 6))

# Subplot A: Weights vs Temperature (Evolution of all 3 components)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(temperatures, pca_weights[:, 0], '-o', label=f'PC1 ({explained_var[0]:.1f}%)')
ax1.plot(temperatures, pca_weights[:, 1], '-s', label=f'PC2 ({explained_var[1]:.1f}%)')
ax1.plot(temperatures, pca_weights[:, 2], '-^', label=f'PC3 ({explained_var[2]:.1f}%)')

ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("PCA Score")
ax1.set_title("Evolution of Top 3 Principal Components")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot B: 3D Trajectory (Phase Space Analysis)
# By mapping the 3 highest Principal Components as a connected path through spatial axes, 
# we construct a topological 'phase space'. Continuous smooth curves in this space indicate 
# smooth parametric shifts (like linear thermal expansion), while abrupt hinges, breaks, 
# or fragmented clusters invariably imply distinct physical mechanisms or totally discontinuous 
# phase transformations taking over.
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc = ax2.scatter(pca_weights[:, 0], 
                 pca_weights[:, 1], 
                 pca_weights[:, 2], 
                 c=temperatures, 
                 cmap='coolwarm', 
                 s=60, 
                 edgecolors='k')

# Connect the dots to show the path
ax2.plot(pca_weights[:, 0], pca_weights[:, 1], pca_weights[:, 2], 'k-', alpha=0.3)

ax2.set_xlabel(f"PC1")
ax2.set_ylabel(f"PC2")
ax2.set_zlabel(f"PC3")
ax2.set_title("3D Phase Trajectory")

# Add colourbar to map thermal progression
cbar = plt.colorbar(sc, ax=ax2, pad=0.1)
cbar.set_label('Temperature (°C)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_PCA_Components_and_Phase_Space.png'), dpi=300)
plt.savefig(os.path.join(output_dir, '04_PCA_Components_and_Phase_Space.svg'))
plt.show()

# =============================================================================
# 7. HIERARCHICAL CLUSTERING ANALYSIS
# =============================================================================
# This approach constructs a quantitative 'lineage' tree of structural relationships.

print("\n--- Running Hierarchical Clustering ---")

# We compute a Linkage Matrix exclusively using Ward's minimal variance method.
# Unlike flat discrete clustering metrics (e.g. K-Means/Spectral), hierarchical 
# clustering visually maps precisely when and how structurally disparate phases branch 
# away from each other mathematically. 
# Lower merges in the dendrogram indicate structures that are functionally identical,
# while high-level splits confidently expose the primary phase partitions dominating 
# the material's bulk composition.
Z = linkage(pdf_curves, method='ward')
# Linkage matrix format: (idx1, idx2, distance, sample_count)
df_linkage = pd.DataFrame(Z, columns=['Cluster 1', 'Cluster 2', 'Distance', 'Sample Count'])
df_linkage.to_csv(os.path.join(output_dir, '05_Hierarchical_Linkage.csv'), index=False)

# Reorder children so temperatures increase monotonically within each branch
Z_sorted = sort_linkage_by_temperature(Z, temperatures)

plt.figure(figsize=(10, 6))
dendrogram(Z_sorted,
           labels=temperatures,
           leaf_rotation=90,
           leaf_font_size=10,
           color_threshold=0.7 * max(Z_sorted[:, 2]))
plt.title("Method 5: Hierarchical Clustering Dendrogram\n"
          "(leaves ordered by temperature within each branch)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Ward's Distance (Dissimilarity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_Hierarchical_Dendrogram.png'), dpi=300)
plt.savefig(os.path.join(output_dir, '05_Hierarchical_Dendrogram.svg'))
plt.close()

print("\nAll analyses complete.")
