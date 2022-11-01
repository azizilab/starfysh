import numpy as np
import pandas as pd
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VisiumDataset(Dataset):
    """
    Loading preprocessed Visium AnnData, gene signature & Anchor spots for Starfysh training
    """

    def __init__(
        self,
        adata,
        gene_sig_exp_m,
        adata_pure,
        library_n
    ):
        spots = adata.obs_names
        genes = adata.var_names
        
        x = adata.X if isinstance(adata.X, np.ndarray) else adata.X.A
        #self.ci = adata.obs['c_i']
        self.expr_mat = pd.DataFrame(x, index=spots, columns=genes)
        #self.gsva = gsva_score
        self.gene_sig_exp_m = gene_sig_exp_m
        self.adata_pure = adata_pure
        self.library_n = library_n

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )

        return (sample, 
                #torch.Tensor(self.ci[idx]), 
                #torch.Tensor(self.gsva.iloc[idx,:]),
                torch.Tensor(self.gene_sig_exp_m.iloc[idx,:]),
                torch.Tensor(self.adata_pure[idx,:]),
                torch.Tensor(self.library_n[idx,None]),
               )


class VisiumPoEDataSet(Dataset):
    """
    return the data stack with expression and image 
    """

    def __init__(
        self,
        train_data,
        map_info,
        patch_r,
        adata_pure,
        library_n,
        gene_sig_exp_m
    ):
        
        super(VisiumPoEDataSet, self).__init__()
        spots = train_data[0].obs_names
        genes = train_data[0].var_names
        
        x = train_data[0].X if isinstance(train_data[0].X, np.ndarray) else train_data[0].X.A
        self.genexp = pd.DataFrame(x, index=spots, columns=genes)
        self.gene_sig_exp_m = gene_sig_exp_m
        self.image = train_data[1]
        self.map_info =  map_info
        self.spot_img_stack = []
        self.adata_pure = adata_pure
        self.library_n = library_n
        
        for i in range(len(self.genexp)):
            img_xmin = int(self.map_info.iloc[i]['imagecol'])-patch_r
            img_xmax = int(self.map_info.iloc[i]['imagecol'])+patch_r
            img_ymin = int(self.map_info.iloc[i]['imagerow'])-patch_r
            img_ymax = int(self.map_info.iloc[i]['imagerow'])+patch_r
            self.spot_img_stack.append(self.image[img_ymin:img_ymax,img_xmin:img_xmax])

    def __len__(self):
        return len(self.genexp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.genexp.iloc[idx, :], dtype='float')
        )
        return (sample,   
                torch.Tensor(self.adata_pure[idx,:]),
                torch.Tensor(self.library_n[idx,None]),
                self.spot_img_stack[idx],
                self.map_info.index[idx],
                torch.Tensor(self.gene_sig_exp_m.iloc[idx,:]),
               )

