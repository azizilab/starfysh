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
                torch.Tensor(self.gexp.iloc[idx, :]),  # normalized signature exprs
                torch.Tensor(self.anchor_idx[idx, :]),  # anchors
                torch.Tensor(self.library_n[idx,None]),  # library size
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
        self.r = args.params['patch_r']
        self.spot_img_stack = []

        assert self.image is not None,\
            "Empty paired H&E image," \
            "please use regular `Starfysh` without PoE integration" \
            "if your dataset doesn't contain histology image"

        # Retrieve image patch around each spot
        scalef = args.scalefactor['tissue_hires_scalef']  # High-res scale factor
        h, w, d = self.image.shape

        for i in range(len(self.expr_mat)):
            xc = int(np.round(self.map_info.iloc[i]['imagecol'] * scalef))
            yc = int(np.round(self.map_info.iloc[i]['imagerow'] * scalef))

            # boundary conditions: edge spots
            yl, yr = max(0, yc-self.r), min(self.image.shape[0], yc+self.r)
            xl, xr = max(0, xc-self.r), min(self.image.shape[1], xc+self.r)
            top = max(0, self.r-yc)
            bottom = h if h > (yc+self.r) else h-(yc+self.r)
            left = max(0, self.r-xc)
            right = w if w > (xc+self.r) else w-(xc+self.r)

            patch = np.zeros((self.r*2, self.r*2, d))
            patch[top:bottom, left:right] = self.image[yl:yr, xl:xr]
            self.spot_img_stack.append(patch)

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

