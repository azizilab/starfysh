<img src=figure/logo.png width="700" />


## Starfysh: Spatial Transcriptomic Analysis using Reference-Free auxiliarY deep generative modeling and Shared Histology

Starfysh is an end-to-end toolbox for analysis and integration of ST datasets. In summary, the Starfysh framework consists of reference-free deconvolution of cell types and cell states, which can be improved with integration of paired histology images of tissues, if available. To facilitate comparison of tissues between healthy or disease contexts and deriving differential spatial patterns, Starfysh is capable of integrating data from multiple tissues and further identifies common or sample-specific spatial “hubs”, defined as neighborhoods with a unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh performs downstream analysis on the spatial organization of hubs and identifies critical genes with spatially varying patterns as well as cell-cell interaction networks. 

To circumvent the need for a single-cell reference in deconvolving cell types, Starfysh leverages two key concepts to determine spots with the most distinct expression profiles as “anchors” that in turn pull apart the remainder of spots: First, Starfysh incorporates a compendium of known cell type marker genesets as well as any additional markers provided by the user. Assuming that the spots with the highest overall expression of a cell type geneset are likely to have the highest proportion of that cell type, these spots form an initial set of anchors. Second, since cell state markers can be context-dependent or not well-characterized, Starfysh utilizes archetypal analysis to refine the initial anchor set and further adds non-overlapping archetypes as additional anchors to enable discovery of novel cell states or a hierarchy of cell states. This feature is particularly useful in characterizing tissue-specific cell states (e.g. patient-specific tumor cell states), their phenotypic transformation in space and associated crosstalk with the microenvironment.

<img src=figure/github_figure_1.png width="1000" />


<img src=figure/github_figure_2.png width="1000" />


## Update
- V 1.0.0 
  - Check the [tutorial on a simple simulated data](https://github.com/azizilab/starfysh/blob/main/notebooks/Starfysh%20tutorial%20on%20a%20toy%20dataset.ipynb)
  
  - Incoming tutorial: 
  
     - with integration of histology
     
     - with an real ST data example
  
  - Check our preprint
  

  
- V 0.0.0 (on Zenodo):  
  - Cite as: He, Siyu, Jin, Yinuo, Nazaret, Achille, Shi, Lingting, Chen, Xueer, & Azizi, Elham. (2022). STARFYSH. Zenodo. [link here](https://doi.org/10.5281/zenodo.6950761)



## Models:
- Semi-supervised learning with Auxiliary Variational Autoencoder (AVAE) for cell-type deconvolution
- Archetypal analysis for unsupervised cell-type discovery (novel cell types) & marker gene refinement (existing annotated cell types)
- Product-of-Experts (PoE) for H&E image integration

- Input:
  - Spatial Transcriptomics count matrix
  - Annotated signature gene sets
  - (Optional): paired H&E image
  
- Output:
  - Spot-wise deconvolution matrix (`q(c)`)
  - Low-dimensional manifold representation (`q(z)`)
  - Clusterings (single-sample) / Hubs (multiple-sample integration) given the deconvolution results
  - Co-localization networks across cell types and Spatial R-L interactions
  - Imputated count matrix (`p(x)`)

## Installation
```bash
# install
python setup.py install --user

# uninstall
pip uninstall starfysh
```

## Features:
- Deconvoluting cell types 

- Generating histology

- Identifying new cell types
Use spatial transcriptomics expression data & annotated signature gene sets as input; perform deconvolution and reconstructure features from the bottle neck neurons; we hope these could capture gene sets representing specific functional modules.


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
sig_path = 'signature/signatures.csv' # specify signature directory
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