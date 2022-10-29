<img src=https://github.com/azizilab/starfysh/blob/main/_figure/logo.png width="800" />


## Starfysh: Spatial Transcriptomic Analysis using Reference-Free auxiliarY deep generative modeling and Shared Histology

<img src=https://github.com/azizilab/starfysh/blob/main/_figure/github_figure_1.png width="1000" />


<img src=https://github.com/azizilab/starfysh/blob/main/_figure/github_figure_2.png width="1000" />

V0.1 (on Zenodo):  Cite as

He, Siyu, Jin, Yinuo, Nazaret, Achille, Shi, Lingting, Chen, Xueer, & Azizi, Elham. (2022). STARFYSH. Zenodo. https://doi.org/10.5281/zenodo.6950761

V0.2 (documents coming soon...)


## Problem setting: 

Spatial Transcriptomics (ST / Visium) data captures gene expressions as well as their locations. However, with limited spatial resolution, each spot usually covers more than 1 cells. To infer potential cellular interactions, we need to infer deconvoluted components specific to each cell-type from the spots to infer functional modules describing cellular states. 



## Models:
- Semi-supervised Autoencoder
Use spatial transcriptomics expression data & annotated signature gene sets as input; perform deconvolution and reconstructure features from the bottle neck neurons; we hope these could capture gene sets representing specific functional modules.


## Directories
```
.
├── data:           Spatial Transcritomics & synthetic simulation datasets
├── notebooks:      Sample notebook & tutorial (to be updated)
├── run_PoE:        Pipeline notebooks to generate pre/post-processing & analysis figures
├── scripts:        Exploratory analysis notebooks & pipeline scripts
├── semiVAE_all:    Combined model ( i). expression-based deconvolution; ii). expression + image (PoE) deconvolution
├── simulation:     Synthetic simulation from scRNA-seq for benchmark
```
