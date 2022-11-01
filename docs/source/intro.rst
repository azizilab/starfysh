Overview
========
Starfysh is an end-to-end toolbox for analysis and integration of ST datasets.
In summary, the Starfysh framework consists of reference-free deconvolution of cell types and cell states, which can be improved with integration of paired histology images of tissues, if available. To facilitate comparison of tissues between healthy or disease contexts and deriving differential spatial patterns, Starfysh is capable of integrating data from multiple tissues and further identifies common or sample-specific spatial “hubs”, defined as neighborhoods with a unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh performs downstream analysis on the spatial organization of hubs and identifies critical genes with spatially varying patterns as well as cell-cell interaction networks.


Features
********
Use spatial transcriptomics expression data & annotated signature gene sets as input; perform deconvolution and reconstructure features from the bottle neck neurons; we hope these could capture gene sets representing specific functional modules.

- Deconvoluting cell types

- Generating histology

- Identifying new cell types



Model Specification
*******************
Starfysh is based on semi-supervised learning with Auxiliary Variational Autoencoder (AVAE) for cell-type deconvolution
. We further applied Archetypal analysis for unsupervised cell-type discovery & annotation, marker gene identification.
We further apply Product-of-Experts (PoE) for H&E image integration to aid deconvolution with spatial inforation.

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


