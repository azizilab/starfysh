from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scanpy as sc

from torch.autograd import Variable
from torchvision.datasets import MNIST

import pandas as pd
import numpy as np
import json

import torchvision.transforms as transforms

def resize_img(img):
    resize = transforms.Resize((25, 13),interpolation=transforms.InterpolationMode.NEAREST)
    img = resize(img)
    return img

    
def find_data(dat_folder = None,dat_type = None):
    """
    dat_folder
    dat_type: BC, Brain, MidOrg
    """

    if dat_folder == None:
        dat_folder = '/content/drive/MyDrive/SpatialModelProject/data'
        #dat_folder = '../data'

    dat_list = os.listdir(dat_folder)
    
    dat_find = pd.DataFrame(dat_list).loc[pd.DataFrame(dat_list).loc[:,0].str.startswith(str(dat_type)),0]
    return np.array(dat_find)


def load_dat(dat_name = None,
             dat_folder = None,
             var_gene=2000,
             processed = False,
              ):
    """
    load the raw datasets or processed datasets
    for raw datasets, need to specify the var_genes, default: 6000
    return type: anndata, image matrix
    """
    if dat_folder == None:
        dat_folder = '/content/drive/MyDrive/SpatialModelProject/data'

    
    all_file = os.listdir(os.path.join(dat_folder,dat_name))
    find_name = pd.DataFrame(all_file).loc[pd.DataFrame(all_file).loc[:,0].str.endswith('h5'),0].values[0]
    image_path = os.path.join(dat_folder,dat_name,'spatial','.png')



    if processed==False:
        print('loading raw data',dat_name)
        adata = sc.read_visium(path=str(os.path.join(dat_folder,dat_name)), 
                               count_file = find_name, 
                               library_id=str(dat_name),
                               load_images=True, 
                               source_image_path=str(image_path)
                               )
        adata.var_names_make_unique()
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

        #sc.pp.filter_cells(adata, min_counts=2500)
        #sc.pp.filter_cells(adata, max_counts=35000)
        print(f"#cells before MT filter: {adata.n_obs}")
        adata = adata[adata.obs["pct_counts_mt"] < 98]
        print(f"#cells after MT filter: {adata.n_obs}")
        sc.pp.filter_genes(adata, min_cells=10)
        #sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, inplace=False)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=var_gene)
        #adata = adata[:, adata.var.highly_variable]

        adata2 = sc.read_visium(path=str(os.path.join(dat_folder,dat_name)), 
                               count_file = find_name, 
                               library_id=str(dat_name),
                               load_images=True, 
                               source_image_path=str(image_path)
                               )
        adata2.var_names_make_unique()
        adata2.var["mt"] = adata2.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata2, qc_vars=["mt"], inplace=True)
        adata2 = adata2[adata2.obs["pct_counts_mt"] < 98]
        sc.pp.filter_genes(adata2, min_cells=10)
        adata2 = adata2[:, adata.var.highly_variable]

        
    else:
        print('loading processed data',dat_name)
        
        adata2 = sc.read(os.path.join(dat_folder,dat_name+'.h5ad'))
        print(adata2)
        
    return adata2



def prep_dat(adata = None,
             N = 200,
             img_size = 300,
              ):
    """
    Cut the spatial transcriptome and images data into small pieces
    specify the img_size, default: 30 

    return:
    two ndarray: one with expression [N, 30, 30], one with images [N, 30,30]
    N = number of the cutted images
    n_radius = patches are size with [n_radius*2, n_radius*2] 
    """
    library_id = list(adata.uns.get("spatial",{}).keys())[0]
    img = adata.uns['spatial'][str(library_id)]['images']['hires']
    spot_size = adata.uns['spatial'][str(library_id)]['scalefactors']['spot_diameter_fullres']
    scale_factor = adata.uns['spatial'][str(library_id)]['scalefactors']['tissue_hires_scalef']
    circle_radius = scale_factor * spot_size * 0.5

    #ax = plt.axes()
    #ax.imshow(img)
    image_spot = (adata.obsm['spatial'] * scale_factor)
    #ax.scatter(image_spot[:,0],image_spot[:,1])
    
    randon_index=np.random.choice(image_spot.shape[0], size=N, replace=False)
    random_spot = image_spot[randon_index, :]
    #spot_xy_int = (image_spot_xy.astype(int)/10).astype(int)

    #ax.scatter(random_spot[:,0],random_spot[:,1])

    exp_stack = []
    img_stack = []
    exp_stack_info = []
    for i in range(N):
        print('generating patches Number = ',i)
        spot_x = random_spot[i,0]
        spot_y = random_spot[i,1]

        x_index = adata.obs['array_col'][randon_index[i]]
        y_index = adata.obs['array_row'][randon_index[i]]

        spot_in_region_xl,spot_in_region_xr = x_index-12,x_index+12
        spot_in_region_xl,spot_in_region_xr = y_index-6,y_index+6

        exp_i = np.zeros([25,13,adata.n_vars])
        print('exp_i',exp_i.shape)
        exp_info_i = np.zeros([25,13])

        spot_in_region=[]
        for ii in range(exp_i.shape[0]):
            for jj in range(exp_i.shape[1]):
                locate = np.where((adata.obs['array_col']==x_index-12+ii)&(adata.obs['array_row']==y_index-6+jj))
                if locate[0].size>0:
    
                    spot_index = adata.obs.index[locate[0][0]]
                    exp_i[ii,jj] = adata.to_df().loc[spot_index]
                    exp_info_i[ii,jj]=spot_index
                    
                    #spot_in_region.append(image_spot[locate[0],:][0][0])
            
        #spot_in_region = np.array(spot_in_region)

        x_l = int(spot_x-img_size/2)
        x_r = int(spot_x+img_size/2)
        y_l = int(spot_y-img_size/2)
        y_r = int(spot_y+img_size/2)

        img_i = img[x_l:x_r,y_l:y_r,]

        #ax.scatter(spot_in_region[:,0],spot_in_region[:,1])
        #ax.scatter(spot_x,spot_y,s=200)

        #ax.hlines(y=y_l, xmin=x_l, xmax=x_r, linewidth=2, color='r')
        #ax.hlines(y=y_r, xmin=x_l, xmax=x_r, linewidth=2, color='r')
        #ax.vlines(x=x_l, ymin=y_l, ymax=y_r, linewidth=2, color='r')
        #ax.vlines(x=x_r, ymin=y_l, ymax=y_r, linewidth=2, color='r')

        exp_stack.append(exp_i)
        img_stack.append(img_i)
        exp_stack_info.append(exp_info_i)
        #print(exp_info_i)

    return np.array(exp_stack),np.array(img_stack), np.array(exp_stack_info)

def prep_save_dat(avail_dat,dat_folder):

    ## prepare and save datasets
    for dat_name in avail_dat[0:]:
        adata = load_dat(dat_name = dat_name, dat_folder = None, var_gene=2000, processed = False)
        adata.obs.index=np.array([x for x in range(adata.n_obs)])
        print('prep dataset...',dat_name)
        exp_stack, img_stack, exp_stack_info = prep_dat(adata,N=200)
        np.save(os.path.join(dat_folder,dat_name,'exp_stack.npy'),exp_stack)
        np.save(os.path.join(dat_folder,dat_name,'img_stack.npy'),img_stack)
        np.save(os.path.join(dat_folder,dat_name,'exp_stack_info.npy'),exp_stack_info)
        np.save(os.path.join(dat_folder,dat_name,'gene_name.npy'),adata.var.index)

    return _

def load_saved_dat(avail_dat,dat_folder,idx):
    dat_name = avail_dat[idx]

    gene_name = np.load(os.path.join(dat_folder,dat_name,'gene_name.npy'),allow_pickle=True)
    exp_stack = np.load(os.path.join(dat_folder,dat_name,'exp_stack.npy'))
    img_stack = np.load(os.path.join(dat_folder,dat_name,'img_stack.npy'))
    exp_stack_info = np.load(os.path.join(dat_folder,dat_name,'exp_stack_info.npy'))

    return gene_name,exp_stack,img_stack,exp_stack_info

def load_signature():
    cell_signature = pd.read_csv('/content/drive/MyDrive/SpatialModelProject/data/Tcellsignatures_june2020_sele.csv')
    #cell_signature = pd.read_csv('../data/Tcellsignatures_june2020_sele.csv')
    return cell_signature

def one_hot_signature(gene_name, n_type: int, cell_signature) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot_sig = torch.zeros(gene_name.size, n_type)
    for i in range(cell_signature.shape[0]):
        for j in range(cell_signature.shape[1]):
            #print(cell_signature.iloc[i,j])
            #print(gene_name)
            #print(cell_signature.iloc[i,j]==gene_name)
            onehot_sig[cell_signature.iloc[i,j]==gene_name,j]=1.0
    return onehot_sig.type(torch.float32)

def show_histo(histo_img,idx):
    ax = plt.subplot()
    ax.imshow(histo_img.transpose(1,2).transpose(2,3)[idx])
    return ax