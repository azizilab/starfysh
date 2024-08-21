<img src=figure/logo.png width="600" />

[![Documentation Status](https://readthedocs.org/projects/cellpose/badge/?version=latest)](https://readthedocs.org/projects/starfysh/badge/?version=latest)
[![Licence: GPL v3](https://img.shields.io/github/license/azizilab/starfysh)](https://github.com/azizilab/starfysh/blob/master/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a_mxF6Ot5vA_xzr5EaNz-GllYY-pXhwA)

## Starfysh: Spatial Transcriptomic Analysis using Reference-Free deep generative modeling with archetYpes and Shared Histology 

Starfysh is an end-to-end toolbox for the analysis and integration of Spatial Transcriptomic (ST) datasets. In summary, the Starfysh framework enables reference-free deconvolution of cell types and cell states and can be improved with the integration of paired histology images of tissues, if available. Starfysh is capable of integrating data from multiple tissues. In particular, Starfysh identifies common or sample-specific spatial “hubs” with unique composition of cell types. To uncover mechanisms underlying local and long-range communication, Starfysh can be used to perform downstream analysis on the spatial organization of hubs.

<img src=figure/github_figure_1.png width="1000" />

<img src=figure/github_figure_2.png width="1000" />

### Quickstart tutorials
  - [1. Basic deconvolution on an example breast cancer data (dataset & signature files included).](notebooks/Starfysh_tutorial_real.ipynb)
  - [2. Histology integration & deconvolution without pre-defined signatures.](notebooks/Starfysh_tutorial_real_wo_signatures.ipynb) 
  - [3. Multi-sample integration](notebooks/Starfysh_tutorial_integration.ipynb)


Please refer to [Starfysh Documentation](http://starfysh.readthedocs.io) for additional tutorials & APIs

### Installation
Github-version installation:
```bash
# Step 1: Clone the Repository
git clone https://github.com/azizilab/starfysh.git

# Step 2: Navigate to the Repository
cd starfysh

# Step 3: Install the Package
pip install .
```


### Model Input:
  - Spatial Transcriptomics count matrix
  - Annotated signature gene sets (see [example](https://drive.google.com/file/d/1AXWQy_mwzFEKNjAdrJjXuegB3onxJoOM/view?usp=share_link))
  - (Optional): paired H&E image

### Features:
- Deconvolving cell types & discovering novel, unannotated cell states
- Integrating with histology images and multi-sample integration
- Downstream analysis: spatial hub identification, cell-type colocalization networks & receptor-ligand (R-L) interactions

### Directories

```
.
├── data:           Spatial Transcritomics & synthetic simulation datasets
├── notebooks:      Sample tutorial notebooks
├── starfysh:       Starfysh core model
```

### How to cite Starfysh
Please cite [Starfysh paper published in Nature Biotechnology](https://www.nature.com/articles/s41587-024-02173-8#citeas): 
```
He, S., Jin, Y., Nazaret, A. et al.
Starfysh integrates spatial transcriptomic and histologic data to reveal heterogeneous tumor–immune hubs.
Nat Biotechnol (2024).
https://doi.org/10.1038/s41587-024-02173-8
```

### BibTex
```
@article{He2024,
  title = {Starfysh integrates spatial transcriptomic and histologic data to reveal heterogeneous tumor–immune hubs},
  ISSN = {1546-1696},
  url = {http://dx.doi.org/10.1038/s41587-024-02173-8},
  DOI = {10.1038/s41587-024-02173-8},
  journal = {Nature Biotechnology},
  publisher = {Springer Science and Business Media LLC},
  author = {He,  Siyu and Jin,  Yinuo and Nazaret,  Achille and Shi,  Lingting and Chen,  Xueer and Rampersaud,  Sham and Dhillon,  Bahawar S. and Valdez,  Izabella and Friend,  Lauren E. and Fan,  Joy Linyue and Park,  Cameron Y. and Mintz,  Rachel L. and Lao,  Yeh-Hsing and Carrera,  David and Fang,  Kaylee W. and Mehdi,  Kaleem and Rohde,  Madeline and McFaline-Figueroa,  José L. and Blei,  David and Leong,  Kam W. and Rudensky,  Alexander Y. and Plitas,  George and Azizi,  Elham},
  year = {2024},
  month = mar 
}
```

If you have questions, please contact the authors:

- Siyu He - sh3846@columbia.edu
- Yinuo Jin - yj2589@columbia.edu

 
