<img src=figure/logo.png width="600" />

[![Documentation Status](https://readthedocs.org/projects/cellpose/badge/?version=latest)](https://readthedocs.org/projects/starfysh/badge/?version=latest)
[![Licence: GPL v3](https://img.shields.io/github/license/azizilab/starfysh)](https://github.com/azizilab/starfysh/blob/master/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/152y-RpmRTEUJ16c_kF3KRwSRjm_THupv?authuser=1)

## Starfysh: Spatial Transcriptomic Analysis using Reference-Free auxiliarY deep generative modeling and Shared Histology

Starfysh is an end-to-end toolbox for the analysis and integration of Spatial Transcriptomic (ST) datasets. In summary, the Starfysh framework enables reference-free deconvolution of cell types and cell states and can be improved with the integration of paired histology images of tissues, if available. Starfysh is capable of integrating data from multiple tissues. In particular, Starfysh identifies common or sample-specific spatial “hubs” with unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh can be used to perform downstream analysis on the spatial organization of hubs.

<img src=figure/github_figure_1.png width="1000" />

<img src=figure/github_figure_2.png width="1000" />

## Quickstart tutorials on Google Colab
  - [1. Basic deconvolution on an example breast cancer data (dataset & signature files included).](https://colab.research.google.com/drive/152y-RpmRTEUJ16c_kF3KRwSRjm_THupv?authuser=1) 
  - [2. Deconvolution and integration of multiple datasets.]
  - [3. Histology integration & multi-sample integration]

Please refer to [Starfysh Documentation](http://starfysh.readthedocs.io) for additional tutorials & APIs

## Update

- V 1.1.0
  - Simplified visualizations of deconvolution & gene expression predictions
- V 1.0.0
  - [Example dataset](https://drive.google.com/drive/folders/15mK8E0qosELLCFMiDMdPQg8wYcB8mVUv?usp=share_link) & [Zenodo V 1.0.0](https://doi.org/10.5281/zenodo.7342798)


## Installation

```bash
# install
pip install Starfysh
```

## Models & I/O:

- Semi-supervised learning with Auxiliary Variational Autoencoder (AVAE) for cell-type deconvolution
- Input:

  - Spatial Transcriptomics count matrix
  - Annotated signature gene sets (see [example](https://drive.google.com/file/d/1AXWQy_mwzFEKNjAdrJjXuegB3onxJoOM/view?usp=share_link))
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

 
