<img src=figure/logo.png width="700" />

## Starfysh: Spatial Transcriptomic Analysis using Reference-Free auxiliarY deep generative modeling and Shared Histology

Starfysh is an end-to-end toolbox for the analysis and integration of Spatial Transcriptomic (ST) datasets. In summary, the Starfysh framework enables reference-free deconvolution of cell types and cell states and can be improved with the integration of paired histology images of tissues, if available. To facilitate the comparison of tissues between healthy and diseased contexts and the derivation of differential spatial patterns, Starfysh is capable of integrating data from multiple tissues. In particular, Starfysh identifies common or sample-specific spatial “hubs”, defined as neighborhoods with a unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh can be used to perform downstream analysis on the spatial organization of hubs. This analysis includes the identification of critical genes with spatially varying patterns as well as cell-cell interaction networks.

<img src=figure/github_figure_1.png width="1000" />

<img src=figure/github_figure_2.png width="1000" />

## Quickstart of our tutorials on colab
  - [1. Basic deconvlution on an examplary breast cancer data.](https://colab.research.google.com/drive/152y-RpmRTEUJ16c_kF3KRwSRjm_THupv?authuser=1) 
  
  - [2. Deconvlution and integration of multiple datasets.]
  
## Update

- V 1.0.0

  - [Documentation of Starfysh](http://starfysh.readthedocs.io) 
  
  - [Example dataset](https://drive.google.com/drive/folders/15mK8E0qosELLCFMiDMdPQg8wYcB8mVUv?usp=share_link)

  - Additional tutorial (coming soon!):

    - Histology integration
    - Downstream analysis & multi-sample integraion

  - [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2022.11.21.517420v1)
  - [Zenodo V 1.0.0](https://doi.org/10.5281/zenodo.7342798)

## Installation

```bash
# install
python setup.py install --user
```

## Models & I/O:

- Semi-supervised learning with Auxiliary Variational Autoencoder (AVAE) for cell-type deconvolution
- Archetypal analysis for unsupervised cell-type discovery (novel cell types) & marker gene refinement (existing annotated cell types)
- Product-of-Experts (PoE) for H&E image integration

- Input:

  - Spatial Transcriptomics count matrix
  - Annotated signature gene sets (see [example](https://drive.google.com/file/d/1yAfAj7PaFJZph88MwhWNXL5Kx5dKMngZ/view?usp=share_link))
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
sig_path = 'data/bc_signatures_version_1013.csv' # specify signature directory
sample_id = 'CID44971_TNBC'

# --- (a) ST matrix ---
adata, adata_norm = utils.load_adata(
    data_path,
    sample_id,
    n_genes=2000
)

# --- (b) paired H&E image + spots info ---
img, map_info = utils.preprocess_img(
    data_path,
    sample_id,
    adata_index=adata.obs.index,
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
    map_info
)

adata, adata_norm = args.get_adata()

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

# Deconvolution results
deconv_prop = inferences['qc_m'].detach().cpu().numpy()
```

## How to cite Starfysh
Please cite our preprint https://www.biorxiv.org/content/10.1101/2022.11.21.517420v1

### BibTex
```
@article{he2022starfysh,
  title={Starfysh reveals heterogeneous spatial dynamics in the breast tumor microenvironment},
  author={He, Siyu and Jin, Yinuo and Nazaret, Achille and Shi, Lingting and Chen, Xueer and Rampersaud, Sham and Dhillon, Bahawar S and Valdez, Izabella and Friend, Lauren E and Fan, Joy Linyue and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
### Chicago
```
He, Siyu, Yinuo Jin, Achille Nazaret, Lingting Shi, Xueer Chen, Sham Rampersaud, Bahawar S. Dhillon et al. "Starfysh reveals heterogeneous spatial dynamics in the breast tumor microenvironment." bioRxiv (2022).
```

If you have questions, please contact the authors:

- Siyu He - sh3846@columbia.edu
- Yinuo Jin - yj2589@columbia.edu

 
