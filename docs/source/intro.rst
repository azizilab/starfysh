Overview
========
Starfysh is an end-to-end toolbox for analysis and integration of ST datasets.
In summary, the Starfysh framework consists of reference-free deconvolution of cell types and cell states, which can be improved with integration of paired histology images of tissues, if available. To facilitate comparison of tissues between healthy or disease contexts and deriving differential spatial patterns, Starfysh is capable of integrating data from multiple tissues and further identifies common or sample-specific spatial “hubs”, defined as neighborhoods with a unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh performs downstream analysis on the spatial organization of hubs and identifies critical genes with spatially varying patterns as well as cell-cell interaction networks.


Features
********

* Deconvolving cell types / cell states
* Discovering and learning novel cell states
* Integrating with histology images and multi-sample integration
* Downstream analysis: spatial hub identification, cell-type colocalization networks & receptor-ligand (R-L) interactions

Model Specifications
********************

Starfysh performs cell-type deconvolution followed by various downstream analysis to discover spatial interactions in tumor microenvironment.
The core deconvolution model is based on semi-supervised Auxiliary Variational Autoencoder (AVAE). We further provide optional Archetypal Analysis (AA) & Product-of-Experts (PoE) for cell-type annotaation and H&E image integration to further aid deconvolution.
Specifically, Starfysh looks for *anchor spots*, the presumed purest spots with the highest proportion of a given cell type guided by signatures, and further deconvolve the remaining spots. Starfysh provides the following options:

**Base feature**:

* Auxiliary Variational AutoEncoder (AVAE):
  Spot-level deconvolution with expected cell types and corresponding annotated *signature* gene sets (default)

**Optional**:

* Archetypal Analysis (AA):
    If signature is not provided:

    * Unsupervised cell type annotation (if the input *signature* is not provided)

    If signature is provided:

    * Novel cell type / cell state discovery (complementary to known cell types from the *signatures*
    * Refine known marker genes by appending archetype-specific differentially expressed genes, and update anchor spots accordingly

* Product-of-Experts (PoE) integration:
    Multi-modal integrative predictions with *expression* & *histology image* by leverging additional side information (e.g. cell density) from H&E image.


I/O
***
- Input:

  - Spatial Transcriptomics count matrix
  - Annotated signature gene sets (`see example <https://drive.google.com/file/d/1yAfAj7PaFJZph88MwhWNXL5Kx5dKMngZ/view?usp=share_link>`_)
  - (Optional): paired H&E image

- Output:

  - Spot-wise deconvolution matrix (`q(c)`)
  - Low-dimensional manifold representation (`q(z)`)
  - Spatial hubs (in-sample or multiple-sample integration)
  - Co-localization networks across cell types and Spatial receptor-ligand (R-L) interactions
  - Reconstructed count matrix (`p(x)`)


