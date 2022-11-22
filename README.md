<img src=figure/logo.png width="700" />

## Starfysh: Spatial Transcriptomic Analysis using Reference-Free auxiliarY deep generative modeling and Shared Histology

Starfysh is an end-to-end toolbox for the analysis and integration of Spatial Transcriptomic (ST) datasets. In summary, the Starfysh framework enables reference-free deconvolution of cell types and cell states and can be improved with the integration of paired histology images of tissues, if available. To facilitate the comparison of tissues between healthy and diseased contexts and the derivation of differential spatial patterns, Starfysh is capable of integrating data from multiple tissues. In particular, Starfysh identifies common or sample-specific spatial “hubs”, defined as neighborhoods with a unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh can be used to perform downstream analysis on the spatial organization of hubs. This analysis includes the identification of critical genes with spatially varying patterns as well as cell-cell interaction networks.

<img src=figure/github_figure_1.png width="1000" />

<img src=figure/github_figure_2.png width="1000" />

## Update

- V 1.0.0

  - Check out Starfysh [tutorial & documentation](http://starfysh.readthedocs.io) & example [dataset](https://drive.google.com/drive/folders/15mK8E0qosELLCFMiDMdPQg8wYcB8mVUv?usp=share_link)

  - Additional tutorial (coming soon!):

    - Histology integration
    - Downstream analysis & multi-sample integraion

  - Check our preprint (coming soon!)
  - Zenodo V 1.0.0: Siyu He, Yinuo Jin, Achille Nazaret, Lingting Shi, Xueer Chen, Sham Rampersaud, Bahawar Dhillon, Izabella Valdez, Lauren E Friend, Joy Linyue Fan, Cameron Y Park, Rachel Mintz, Yeh-Hsing Lao, David Carrera, Kaylee W Fang, Kaleem Mehdi, Madeline Rohde, José L. McFaline-Figueroa, David Blei, … Elham Azizi. (2022). azizilab/starfysh: STARFYSH (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.7342798 

- V 0.1.0:

  - Zenodo V 0.1.0: He, Siyu, Jin, Yinuo, Nazaret, Achille, Shi, Lingting, Chen, Xueer, & Azizi, Elham. (2022). STARFYSH. Zenodo. 

## Installation

```bash
# install
python setup.py install --user

# uninstall
pip uninstall starfysh
```

## Models & I/O:

- Semi-supervised learning with Auxiliary Variational Autoencoder (AVAE) for cell-type deconvolution
- Archetypal analysis for unsupervised cell-type discovery (novel cell types) & marker gene refinement (existing annotated cell types)
- Product-of-Experts (PoE) for H&E image integration

- Input:

  - Spatial Transcriptomics count matrix
  - Annotated signature gene sets (`see example <https://drive.google.com/file/d/1yAfAj7PaFJZph88MwhWNXL5Kx5dKMngZ/view?usp=share_link>`\_)
  - (Optional): paired H&E image

- Output:

  - Spot-wise deconvolution matrix (`q(c)`)
  - Low-dimensional manifold representation (`q(z)`)
  - Spatial hubs (in-sample or multiple-sample integration)
  - Co-localization networks across cell types and Spatial receptor-ligand (R-L) interactions
  - Reconstructed count matrix (`p(x)`)

## Features:

- Deconvolving cell types / cell states

- Discovering and learning novel cell states

- Integrating with histology images and multi-sample integration

- Downstream analysis: spatial hub identification, cell-type colocalization networks & receptor-ligand (R-L) interactions

## Directories

```
.
├── data:           Spatial Transcritomics & synthetic simulation datasets
├── notebooks:      Sample notebook & tutorial
├── simulation:     Synthetic simulation from scRNA-seq for benchmark
├── starfysh:       Starfysh core model
```

## Quickstart

```python
import numpy as np
import pandas as pd
import torch
from starfysh import (
    AA,
    dataloader,
    starfysh,
    utils,
    plot_utils,
    post_analysis
)

# (1) Loading dataset & signature gene sets
data_path = 'data/' # specify data directory
sig_path = 'data/tnbc_signatures.csv' # specify signature directory
sample_id = 'CID44971_TNBC'

# --- (a) ST matrix ---
adata, adata_norm = utils.load_adata(
    data_path,
    sample_id,
    n_genes=2000
)

# --- (b) paired H&E image + spots info ---
hist_img, map_info = utils.preprocess_img(
    data_path,
    sample_id,
    adata.obs.index,
    hchannal=False
)

# --- (c) signature gene sets ---
gene_sig = utils.filter_gene_sig(
    pd.read_csv(sig_path),
    adata.to_df()
)

# (2) Starfysh deconvolution

# --- (a) Preparing arguments for model training
args = utils.VisiumArguments(
    adata,
    adata_norm,
    gene_sig,
    map_info,
    n_anchors=60, # number of anchor spots per cell-type
    window_size=5  # library size smoothing radius
)

adata, adata_noprm = args.get_adata()

# --- (b) Model training ---
n_restarts = 3
epochs = 100
patience = 10
device = torch.device('cpu')

model, loss = utils.run_starfysh(
    args,
    n_restarts,
    epochs=epochs,
    patience=patience
)

# (3). Parse deconvolution outputs
inferences, generatives, px = starfysh.model_eval(
    model,
    adata,
    args.sig_mean,
    device,
    args.log_lib,
)
```
