import numpy as np
import pandas as pd
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# from torchvision import transforms


class VisiumDataset(Dataset):
    """
    Loading preprocessed Visium AnnData, gene signature & Anchor spots for Starfysh training
    """

    def __init__(
        self,
        adata,
        args,
    ):
        spots = adata.obs_names
        genes = adata.var_names

        
        x = adata.X if isinstance(adata.X, np.ndarray) else adata.X.A
        self.expr_mat = pd.DataFrame(x, index=spots, columns=genes)
        self.gexp = args.sig_mean_znorm
        self.anchor_idx = args.pure_idx
        self.library_n = args.win_loglib

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )

        return (sample,
                torch.Tensor(self.gexp.iloc[idx, :]),
                torch.Tensor(self.anchor_idx[idx, :]),
                torch.Tensor(self.library_n[idx,None]),
               )


class VisiumPoEDataSet(VisiumDataset):
    """
    return the data stack with expression and image 
    """

    def __init__(
        self,
        adata,
        args,
    ):

        super(VisiumPoEDataSet, self).__init__(adata, args)
        self.image = args.img
        self.map_info = args.map_info
        self.patch_r = args.params['patch_r']
        self.spot_img_stack = []

        assert self.image is not None,\
            "Empty paired H&E image," \
            "please use regular `Starfysh` without PoE integration" \
            "if your dataset doesn't contain histology image"

        for i in range(len(self.expr_mat)):
            img_xmin = int(self.map_info.iloc[i]['imagecol'])-self.patch_r
            img_xmax = int(self.map_info.iloc[i]['imagecol'])+self.patch_r
            img_ymin = int(self.map_info.iloc[i]['imagerow'])-self.patch_r
            img_ymax = int(self.map_info.iloc[i]['imagerow'])+self.patch_r
            self.spot_img_stack.append(self.image[img_ymin:img_ymax,img_xmin:img_xmax])

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )
        return (sample,   
                torch.Tensor(self.anchor_idx[idx, :]),
                torch.Tensor(self.library_n[idx, None]),
                self.spot_img_stack[idx],
                self.map_info.index[idx],
                torch.Tensor(self.gexp.iloc[idx, :]),
               )

