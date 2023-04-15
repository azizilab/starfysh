import numpy as np
import scipy.stats as stats
import pandas as pd
import scanpy as sc
import anndata
import os
import sys

def load_adata(dat_folder,dat_name,use_other_gene,other_gene_list,n_genes):
    """
    load the visium data.
    dat_folder: the path of the dataset
    dat_name: the name of the dataset
    use_other_gene: whether to use other gene list
    other_gene_list: if True use_other_gene, providing the list
    n_genes: specify the number of the gene to be left

    return: 
    anndata and the gene list used
    
    """
    # find the h5_file
    if dat_name.startswith('MBC'):
        
        find_h5 = pd.DataFrame(os.listdir(os.path.join(dat_folder,dat_name))).loc[pd.DataFrame(os.listdir(os.path.join(dat_folder,dat_name))).loc[:,0].str.endswith('h5'),0].values[0]

        adata = sc.read_visium(path=str(os.path.join(dat_folder,dat_name)), count_file = find_h5, library_id=str(dat_name),load_images=True, source_image_path=str(os.path.join(dat_folder,dat_name,'spatial')))
        adata.var_names_make_unique()
    else:
        find_h5 = pd.DataFrame(os.listdir(os.path.join(dat_folder,dat_name))).loc[pd.DataFrame(os.listdir(os.path.join(dat_folder,dat_name))).loc[:,0].str.endswith('h5ad'),0].values[0]
        adata = sc.read_h5ad(os.path.join(dat_folder, dat_name, dat_name + '.h5ad'))
        adata.var_names_make_unique()

  

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    print('The datasets have',adata.n_obs,'spots, and',adata.n_vars,'genes')
    adata.var['rp'] = adata.var_names.str.startswith('RPS') + adata.var_names.str.startswith('RPL')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['rp'], percent_top=None, log1p=False, inplace=True)

    # Remove mitochondrial genes
    print('removing mt/rp genes')
    adata = adata[:,-adata.var['mt']]
    adata = adata[:,-adata.var['rp']]

    if use_other_gene:
        other_list_temp = other_gene_list.intersection(adata.var.index)
        adata = adata[:,other_list_temp]
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        variable_gene  = other_gene_list
    else:
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_genes,inplace=True)
        adata.raw = adata
        variable_gene = adata.var.highly_variable
        adata = adata[:, variable_gene]
        variable_gene = variable_gene[variable_gene==True].index

  #sc.pp.filter_cells(adata, min_counts=20)
  #sc.pp.filter_cells(adata, max_counts=50000)
  #adata = adata[adata.obs["pct_counts_mt"] < 30]
  #print(f"#cells after MT filter: {adata.n_obs}")


    return adata,variable_gene