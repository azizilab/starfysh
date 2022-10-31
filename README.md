<img src=https://github.com/azizilab/starfysh/blob/main/_figure/logo.png width="800" />


## Starfysh: Spatial Transcriptomic Analysis using Reference-Free auxiliarY deep generative modeling and Shared Histology

Starfysh is an end-to-end toolbox for analysis and integration of ST datasets. In summary, the Starfysh framework consists of reference-free deconvolution of cell types and cell states, which can be improved with integration of paired histology images of tissues, if available. To facilitate comparison of tissues between healthy or disease contexts and deriving differential spatial patterns, Starfysh is capable of integrating data from multiple tissues and further identifies common or sample-specific spatial “hubs”, defined as neighborhoods with a unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh performs downstream analysis on the spatial organization of hubs and identifies critical genes with spatially varying patterns as well as cell-cell interaction networks. 

To circumvent the need for a single-cell reference in deconvolving cell types, Starfysh leverages two key concepts to determine spots with the most distinct expression profiles as “anchors” that in turn pull apart the remainder of spots: First, Starfysh incorporates a compendium of known cell type marker genesets as well as any additional markers provided by the user. Assuming that the spots with the highest overall expression of a cell type geneset are likely to have the highest proportion of that cell type, these spots form an initial set of anchors. Second, since cell state markers can be context-dependent or not well-characterized, Starfysh utilizes archetypal analysis to refine the initial anchor set and further adds non-overlapping archetypes as additional anchors to enable discovery of novel cell states or a hierarchy of cell states. This feature is particularly useful in characterizing tissue-specific cell states (e.g. patient-specific tumor cell states), their phenotypic transformation in space and associated crosstalk with the microenvironment.

<img src=https://github.com/azizilab/starfysh/blob/main/_figure/github_figure_1.png width="1000" />


<img src=https://github.com/azizilab/starfysh/blob/main/_figure/github_figure_2.png width="1000" />




## Updata

- V 1.0.0 (Tutorial coming soon...)
  Check our preprint
  
- V 0.0.0 (on Zenodo):  Cite as

He, Siyu, Jin, Yinuo, Nazaret, Achille, Shi, Lingting, Chen, Xueer, & Azizi, Elham. (2022). STARFYSH. Zenodo. https://doi.org/10.5281/zenodo.6950761



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

## Installation:
Starfysh can be easily installed from pip:


## Functions:
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

## Installation
(To be updated: currently only contain expression-based deconvolution model)
```bash
# install
python setup.py install --user

# uninstall
pip uninstall bcvae

# re-install
./reinstall.sh
```
