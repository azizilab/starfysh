import os
import cv2
import json
import numpy as np
import pandas as pd
import scanpy as sc
# import logging

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from torch.utils.data import DataLoader

import sys
# import histomicstk as htk
from skimage import io

# Module import
from starfysh import LOGGER
from .dataloader import VisiumDataset, VisiumPoEDataSet
from .starfysh import AVAE, AVAE_PoE, train, train_poe


# -------------------
# Model Parameters
# -------------------

class VisiumArguments:
    """
    Loading Visium AnnData, perform preprocessing, library-size smoothing & Anchor spot detection

    Parameters
    ----------
    adata : AnnData
        annotated visium count matrix

    adata_norm : AnnData
        annotated visium count matrix after normalization & log-transform

    gene_sig : pd.DataFrame
        list of signature genes for each cell type. (dim: [S, Cell_type])

    img_metadata : dict
        Spatial information metadata (histology image, coordinates, scalefactor)
    """

    def __init__(
        self,
        adata,
        adata_norm,
        gene_sig,
        img_metadata,
        **kwargs
    ):

        self.adata = adata
        self.adata_norm = adata_norm
        self.gene_sig = gene_sig
        self.map_info = img_metadata['map_info']
        self.img = img_metadata['img']
        self.img_patches = None 
        self.scalefactor = img_metadata['scalefactor']

        self.params = {
            'sample_id': 'ST', 
            'n_anchors': 60,
            'patch_r': 13,
            'vlow': 10,
            'vhigh': 95,
            'sig_version': 'raw',
            'window_size': 30,
            'z_axis': 0,
        }

        # Update parameters for library smoothing & anchor spot identification
        for k, v in kwargs.items():
            if k in self.params.keys():
                self.params[k] = v

        # Update spatial information to adata if missing
        if 'spatial' not in adata.uns_keys() and (self.img is not None and self.scalefactor is not None):
            self._update_spatial_info(self.params['sample_id'])

        # Filter out signature genes X listed in expression matrix
        LOGGER.info('Subsetting highly variable & signature genes ...')
        self.adata = get_adata_wsig(adata, gene_sig)
        self.adata_norm = get_adata_wsig(adata_norm, gene_sig)

        # Get smoothed library size
        LOGGER.info('Smoothing library size by taking averaging with neighbor spots...')
        log_lib = np.log1p(self.adata.X.sum(1))
        self.log_lib = np.squeeze(np.asarray(log_lib)) if log_lib.ndim > 1 else log_lib
        self.win_loglib = get_windowed_library(self.adata,
                                               self.map_info,
                                               self.log_lib,
                                               window_size=self.params['window_size']
                                               )

        # Retrieve & Z-norm signature gexp
        LOGGER.info('Retrieving & normalizing signature gene expressions...')
        self.sig_mean = self._get_sig_mean()
        self.sig_mean_znorm = self._znorm_sig(z_axis=self.params['z_axis'])

        # Get anchor spots
        LOGGER.info('Identifying anchor spots (highly expression of specific cell-type signatures)...')
        anchor_info = get_anchor_spots(self.adata,
                                       self.sig_mean_znorm,
                                       v_low=self.params['vlow'],
                                       v_high=self.params['vhigh'],
                                       n_anchor=self.params['n_anchors']
                                       )
        self.pure_spots, self.pure_dict, self.pure_idx = anchor_info
        self.alpha_min = get_alpha_min(self.sig_mean, self.pure_dict)  # Calculate alpha mean

    def get_adata(self):
        """Return adata after preprocessing & HVG gene selection"""
        return self.adata, self.adata_norm

    def get_anchors(self):
        """Return indices of anchor spots for each cell type"""
        anchors_df = pd.DataFrame.from_dict(self.pure_dict, orient='columns')
        return anchors_df.applymap(
            lambda x:
            np.where(self.adata.obs.index == x)[0][0] # TODO: make sure adata.obs index is formatted as "location_i"
        )

    def get_img_patches(self):
        assert self.img_patches is not None, "Please run Starfysh PoE first"
        return self.img_patches

    def update_anchors(self, gene_sig_updated):
        """Re-calculate anchor spots given updated gene signatures"""
        self.gene_sig = gene_sig_updated
        self.sig_mean = self._get_sig_mean()
        self.sig_mean_znorm = self._znorm_sig(z_axis=self.params['z_axis'])

        LOGGER.info('Recalculating anchor spots (highly expression of specific cell-type signatures)...')
        anchor_info = get_anchor_spots(self.adata,
                                       self.sig_mean_znorm,
                                       v_low=self.params['vlow'],
                                       v_high=self.params['vhigh'],
                                       n_anchor=self.params['n_anchors']
                                       )
        self.pure_spots, self.pure_dict, self.pure_idx = anchor_info

    def _get_sig_mean(self):
        gene_sig_exp_m = pd.DataFrame()
        adata_df = self.adata.to_df()
        for i in range(self.gene_sig.shape[1]):
            # calculate avg. signature expressions from raw count
            if self.params['sig_version'] == 'raw':
                gene_sig_exp_m[self.gene_sig.columns[i]] = np.nanmean((
                    adata_df.loc[
                    :,
                    np.intersect1d(
                        self.adata.var_names,
                        np.unique(self.gene_sig.iloc[:, i].astype(str))
                    )
                    ]
                ), axis=1)

            # calculate avg. signature  expressions from log count
            else:
                gene_sig_exp_m[self.gene_sig.columns[i]] = adata_df.loc[
                                                           :,
                                                           np.intersect1d(
                                                               self.adata.var_names,
                                                               np.unique(self.gene_sig.iloc[:, i].astype(str))
                                                           )
                                                           ].mean(axis=1)

        gene_sig_exp_arr = pd.DataFrame(np.array(gene_sig_exp_m), columns=gene_sig_exp_m.columns,
                                        index=self.adata.obs_names)
        return gene_sig_exp_arr

    def _update_spatial_info(self, sample_id):
        """Update paired spatial information to ST adata"""
        self.adata.uns['spatial'] = {
            sample_id: {
                'images': {'hires': (self.img - self.img.min()) / (self.img.max() - self.img.min())},
                'scalefactors': self.scalefactor
            },
        }

        self.adata_norm.uns['spatial'] = {
            sample_id: {
                'images': {'hires': (self.img - self.img.min()) / (self.img.max() - self.img.min())},
                'scalefactors': self.scalefactor
            },
        }

        self.adata.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values
        self.adata_norm.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values
        return None

    def _update_img_patches(self, dl_poe):
        imgs = torch.Tensor(dl_poe.spot_img_stack)
        self.img_patches = imgs.reshape(imgs.shape[0], -1)
        return None

    def _znorm_sig(self, z_axis, eps=1e-12):
        """Z-normalize average expressions for each gene"""
        sig_mean = self.sig_mean + eps
        sig_mean_zscore = sig_mean.apply(zscore, axis=z_axis)
        sig_mean_zscore[sig_mean_zscore < 0] = 0
        sig_mean_zscore = sig_mean_zscore.fillna(0)
        return sig_mean_zscore


# --------------------------------
# Running starfysh with 3-restart
# --------------------------------

def init_weights(module):
    if type(module) == nn.Linear:
        # torch.nn.init.xavier_normal(module.weight)
        # torch.nn.init.kaiming_normal_(module.weight)
        torch.nn.init.kaiming_uniform_(module.weight)

    elif type(module) == nn.BatchNorm1d:
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def run_starfysh(
        visium_args,
        n_repeats=3,
        lr=0.001,
        epochs=100,
        patience=10,
        poe=False,
        device=torch.device('cpu'),
        verbose=True
):
    """
    Wrapper to run starfysh deconvolution.

        Note: adding early-stopping mechanism evaluated by loss c
        - early-stopping with patience=10
        - choose best among 3 rerun

    Parameters
    ----------
    visium_args : VisiumArguments
        Preprocessed metadata calculated from input visium matrix:
        e.g. mean signature expression, library size, anchor spots, etc.

    n_repeats : int
        Number of restart to run Starfysh

    epochs : int
        Max. number of iterations

    patience : int
        Max. counts for early-stopping if q(c) doesn't drop

    poe : bool
        Whether to perform inference with Poe w/ image integration

    Returns
    -------
    best_model : starfysh.AVAE or starfysh.AVAE_PoE
        Trained Starfysh model with deconvolution results

    loss : np.ndarray
        Training losses
    """

    np.random.seed(0)
    max_patience = patience

    # Loading parameters
    adata = visium_args.adata
    win_loglib = visium_args.win_loglib
    gene_sig, sig_mean_znorm = visium_args.gene_sig, visium_args.sig_mean_znorm

    models = [None] * n_repeats
    losses = []
    loss_c_list = np.repeat(np.inf, n_repeats)

    if poe:
        dl_func = VisiumPoEDataSet  # dataloader
        train_func = train_poe  # training wrapper
    else:
        dl_func = VisiumDataset
        train_func = train

    trainset = dl_func(adata=adata, args=visium_args)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    # Running Starfysh with multiple starts
    LOGGER.info('Running Starfysh with {} restarts, choose the model with best parameters...'.format(n_repeats))
    for i in range(n_repeats):
        if verbose:
            LOGGER.info(" ===  Restart Starfysh {0} === \n".format(i + 1))

        patience = max_patience
        best_loss_c = np.inf

        if poe:
            model = AVAE_PoE(
                adata=adata,
                gene_sig=gene_sig,
                patch_r=visium_args.params['patch_r'],
                win_loglib=win_loglib
            )
            # Update patched & flattened image patches
            visium_args._update_img_patches(trainset)
        else:
            model = AVAE(
                adata=adata,
                gene_sig=gene_sig,
                win_loglib=win_loglib
            )
            
        model = model.to(device)

        loss_dict = {
            'reconst': [],
            'c': [],
            'z': [],
            'n': [],
            'tot': []
        }

        # Initialize model params
        if verbose:
            LOGGER.info('Initializing model parameters...')
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            if patience == 0:
                loss_c_list[i] = best_loss_c
                if verbose:
                    LOGGER.info('Saving best-performing model; Early stopping...')
                break

            result = train_func(model, trainloader, device, optimizer)
            torch.cuda.empty_cache()
            loss_tot, loss_reconst, loss_z, loss_c, loss_n, corr_list = result

            if loss_c < best_loss_c:
                patience = max_patience
                models[i] = model
                best_loss_c = loss_c
            else:
                patience -= 1

            torch.cuda.empty_cache()

            loss_dict['tot'].append(loss_tot)
            loss_dict['reconst'].append(loss_reconst)
            loss_dict['z'].append(loss_z)
            loss_dict['c'].append(loss_c)
            loss_dict['n'].append(loss_n)

            if (epoch + 1) % 10 == 0 or patience == 0:
                if verbose:
                    LOGGER.info("Epoch[{}/{}], train_loss: {:.4f}, train_reconst: {:.4f}, train_z: {:.4f},train_c: {:.4f},train_n: {:.4f}".format(
                        epoch + 1, epochs, loss_tot, loss_reconst, loss_z, loss_c, loss_n)
                    )

        losses.append(loss_dict)

        if verbose:
            LOGGER.info(" === Finished training === \n")

    idx = np.argmin(loss_c_list)
    best_model = models[idx]
    loss = losses[idx]

    return best_model, loss


# -------------------
# Preprocessing & IO
# -------------------

def get_alpha_min(sig_mean, pure_dict):
    """Calculate alpha_min for Dirichlet dist. for each factor"""
    alpha_min = 0
    for col_idx in sig_mean.columns:
        if (1 / (sig_mean.loc[pure_dict[col_idx], :] / sig_mean.loc[pure_dict[col_idx], :].sum())[col_idx]).max() > alpha_min:
            alpha_min = (1 / (sig_mean.loc[pure_dict[col_idx], :] / sig_mean.loc[pure_dict[col_idx], :].sum())[col_idx]).max()
    return alpha_min


def preprocess(adata_raw,
               lognorm=True,
               min_perc=None,
               max_perc=None,
               n_top_genes=6000,
               mt_thld=100,
               verbose=True,
               multiple_data=False
               ):
    """
    Preprocessing ST gexp matrix, remove Ribosomal & Mitochondrial genes

    Parameters
    ----------
    adata_raw : annData
        Spot x Bene raw expression matrix [S x G]

    min_perc : float
        lower-bound percentile of non-zero gexps for filtering spots

    max_perc : float
        upper-bound percentile of non-zero gexps for filtering spots

    n_top_genes: float
        number of the variable genes

    mt_thld : float
        max. percentage of mitochondrial gexps for filtering spots
        with excessive MT expressions

    multiple_data: bool
        whether the study need integrate datasets
    """
    adata = adata_raw.copy()

    if min_perc and max_perc:
        assert 0 < min_perc < max_perc < 100, \
            "Invalid thresholds for cells: {0}, {1}".format(min_perc, max_perc)
        min_counts = np.percentile(adata.obs['total_counts'], min_perc)
        sc.pp.filter_cells(adata, min_counts=min_counts)

    # Remove cells with excessive MT expressions
    # Remove MT & RB genes

    if verbose:
        LOGGER.info('Preprocessing1: delete the mt and rp')
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['rb'] = np.logical_or(
        adata.var_names.str.startswith('RPS'),
        adata.var_names.str.startswith('RPL')
    )

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    mask_cell = adata.obs['pct_counts_mt'] < mt_thld
    mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])

    adata = adata[mask_cell, mask_gene]
    sc.pp.filter_genes(adata, min_cells=1)

    if lognorm:
        if verbose:
            LOGGER.info('Preprocessing2: Normalize')
        if multiple_data:
            sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
        else:
            sc.pp.normalize_total(adata, inplace=True)

        # Preprocessing3: Logarithm
        if verbose:
            LOGGER.info('Preprocessing3: Logarithm')
        sc.pp.log1p(adata)
    else:
        if verbose:
            LOGGER.info('Skip Normalize and Logarithm')

    # Preprocessing4: Find the variable genes
    if verbose:
        LOGGER.info('Preprocessing4: Find the variable genes')
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes, inplace=True)

    return adata


def load_adata(data_folder, sample_id, n_genes, multiple_data=False):
    """
    load visium adata with raw counts, preprocess & extract highly variable genes

    Parameters
    ----------
        data_folder : str
            Root directory of the data

        sample_id : str
            Sample subdirectory under `data_folder`

        n_genes : int
            the number of the gene for training

        multiple_data: bool
            whether the study include multiple datasets

    Returns
    -------
        adata : sc.AnnData
            Processed ST raw counts

        adata_norm : sc.AnnData
            Processed ST normalized & log-transformed data
    """
    has_feature_h5 = os.path.isfile(
        os.path.join(data_folder, sample_id, 'filtered_feature_bc_matrix.h5')
    ) # whether dataset stored in h5 with spatial info.

    if has_feature_h5:
        adata = sc.read_visium(path=os.path.join(data_folder, sample_id), library_id=sample_id)
        adata.var_names_make_unique()
        adata.obs['sample'] = sample_id
    elif sample_id.startswith('simu'): # simulations
        adata = sc.read_csv(os.path.join(data_folder, sample_id, 'counts.st_synth.csv'))
    else:
        filenames = [
            f[:-5] for f in os.listdir(os.path.join(data_folder, sample_id))
            if f[-5:] == '.h5ad'
        ]
        assert len(filenames) == 1, \
            "None or more than `h5ad` file in the data directory," \
            "please contain only 1 target ST file in the given directory"
        adata = sc.read_h5ad(os.path.join(data_folder, sample_id, filenames[0] + '.h5ad'))
        adata.var_names_make_unique()
        adata.obs['sample'] = sample_id
    adata_norm = preprocess(adata, n_top_genes=n_genes,multiple_data=multiple_data)
    adata = adata[:, list(adata_norm.var_names)]
    adata.var['highly_variable'] = adata_norm.var['highly_variable']
    adata.obs = adata_norm.obs

    if '_index' in adata.var.columns:
        adata.var_names = adata.var['_index']

    return adata, adata_norm


def load_signatures(filename, adata):
    """
    load annotated signature gene sets

    Parameters
    ----------
    filename : str
        Signature file

    adata : sc.AnnData
        ST count matrix

    Returns
    -------
    gene_sig : pd.DataFrame
        signatures per cell type / state
    """
    assert os.path.isfile(filename), "Unable to find the signature file"
    gene_sig = pd.read_csv(filename, index_col=0)
    gene_sig = filter_gene_sig(gene_sig, adata.to_df())
    sigs = np.unique(
        gene_sig.apply(
            lambda x:
            pd.unique(x[~pd.isna(x)])
        ).values
    )

    return gene_sig, np.unique(sigs)


def preprocess_img(
        data_path,
        sample_id,
        adata_index,
        hchannel=False
):
    """
    Load and preprocess visium paired H&E image & spatial coords

    Parameters
    ----------
    data_path : str
        Root directory of the data

    sample_id : str
        Sample subdirectory under `data_path`

    hchannel : bool
        Whether to apply binary color deconvolution to extract hematoxylin channel
        Please refer to:
        https://digitalslidearchive.github.io/HistomicsTK/examples/color_deconvolution.html

    Returns
    -------
    adata_image : np.ndarray
        Processed histology image

    map_info : np.ndarray
        Spatial coords of spots (dim: [S, 2])
    """
    if os.path.isfile(os.path.join(data_path, sample_id, 'spatial', 'tissue_hires_image.png')):
        if hchannel:
            adata_image = io.imread(
                os.path.join(
                    data_path, sample_id, 'spatial', 'tissue_hires_image.png'
                )
            )

            # adata_image = (adata_image-adata_image.min())/(adata_image.max()-adata_image.min())
            adata_image_norm = (adata_image * 255).astype(np.uint8)
            stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
            # specify stains of input image
            stains = ['hematoxylin',  # nuclei stain
                      'eosin',  # cytoplasm stain
                      'null']  # set to null if input contains only two stains
            # create stain matrix
            W = np.array([stain_color_map[st] for st in stains]).T

            # perform standard color deconvolution
            imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(adata_image_norm, W)

            adata_image_h = imDeconvolved.Stains[:,:,0]
            adata_image_e = imDeconvolved.Stains[:,:,2]

            adata_image_h = ((adata_image_h - adata_image_h.min()) / (adata_image_h.max()-adata_image_h.min()) *255).astype(np.uint8)
            adata_image_e = ((adata_image_e - adata_image_e.min()) / (adata_image_e.max()-adata_image_e.min()) *255).astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

            adata_image_h = clahe.apply(adata_image_h)
            adata_image_e = clahe.apply(adata_image_e)

            adata_image_h = (adata_image_h - adata_image_h.min()) / (adata_image_h.max() - adata_image_h.min())
            adata_image_e = (adata_image_e - adata_image_e.min()) / (adata_image_e.max() - adata_image_e.min())

        else:
            adata_image = io.imread(os.path.join(data_path, sample_id, 'spatial', 'tissue_hires_image.png'))
            # adata_image = (adata_image-adata_image.min())/(adata_image.max()-adata_image.min())
    else:
        adata_image = None

    # Mapping images to location
    f = open(os.path.join(data_path, sample_id, 'spatial', 'scalefactors_json.json', ))
    json_info = json.load(f)
    f.close()
    tissue_hires_scalef = json_info['tissue_hires_scalef']

    tissue_position_list = pd.read_csv(os.path.join(data_path, sample_id, 'spatial', 'tissue_positions_list.csv'), header=None, index_col=0)
    tissue_position_list = tissue_position_list.loc[adata_index, :]
    map_info = tissue_position_list.iloc[:, 1:3]
    map_info.columns = ['array_row', 'array_col']
    map_info.loc[:, 'imagerow'] = tissue_position_list.iloc[:, -2] * tissue_hires_scalef
    map_info.loc[:, 'imagecol'] = tissue_position_list.iloc[:, -1] * tissue_hires_scalef
    map_info.loc[:, 'sample'] = sample_id

    return {
        'img': adata_image,
        'map_info': map_info,
        'scalefactor': json_info
    }


def gene_for_train(adata, df_sig, verbose=True):
    """find the varibale gene name, the mattered gene signatures, and the combined variable+signature for training"""

    variable_gene = adata.var_names[adata.var['highly_variable']]
    if verbose:
        print('\t The number of original variable genes in the dataset', variable_gene.shape)
        print('\t The number of siganture genes in the dataset', np.unique(df_sig.values.flatten().astype(str)).shape)

    # filter out some genes in the signature not in the var_names
    sig_gname_filtered = np.intersect1d(adata.var_names, np.unique(df_sig.values.flatten().astype(str)))
    if verbose:
        print('\t After filter out some genes in the signature not in the var_names ...', sig_gname_filtered.shape)

    # filter out some genes not highly expressed in the signature
    sig_gname_filtered = sig_gname_filtered[adata.to_df().loc[:, sig_gname_filtered].sum() > 0]
    if verbose:
        print('\t After filter out some genes not highly expressed in the signature ...', sig_gname_filtered.shape)

    sig_variable_gene_inter = {*np.array(variable_gene), *sig_gname_filtered}
    if verbose:
        print('\t Combine the varibale and siganture, the total unique gene number is ...', len(sig_variable_gene_inter))

    return list(variable_gene), list(sig_gname_filtered), list(sig_variable_gene_inter)


def get_adata_wsig(adata, gene_sig):
    """
    Select intersection of HVGs from dataset & signature annotations
    """
    unique_sigs = np.unique(np.hstack(
        gene_sig.apply(
            lambda x:
            pd.unique(x[~pd.isna(x)])
        )
    ))
    genes_to_keep = np.intersect1d(
        np.union1d(adata.var_names[adata.var.highly_variable], unique_sigs),
        adata.var_names
    )
    return adata[:, genes_to_keep]


def filter_gene_sig(gene_sig, adata_df):
    for i in range(gene_sig.shape[0]):
        for j in range(gene_sig.shape[1]):
            gene = gene_sig.iloc[i, j]
            if gene in adata_df.columns:
                # We don't filter signature genes based on expression level (prev: threshold=20)
                if adata_df.loc[:, gene].sum() < 0:
                    gene_sig.iloc[i, j] = 'NaN'
    return gene_sig


def get_anchor_spots(
        adata_sample,
        sig_mean,
        v_low=20,
        v_high=95,
        n_anchor=40
):
    """
    Calculate the top `anchor spot` enriched for the given cell type
    (determined by normalized expression values from each signature)

    Parameters
    ----------
        adata_sample: sc.Anndata
            ST raw count

        v_low : int
            the low threshold to filter high-quality spots

        v_high: int
            the high threshold to filter high-quality spots

        n_anchor: int
            # anchor spots per cell type

    Returns
    -------
    pure_spots : np.ndarray
        anchor spot indices per cell type (dim: [S, n_anchor])

    pre_dict : dict
        Cell-type -> Anchor spots

    adata_pure : np.ndarray
        Binary indicators of anchor spots (dim: [S, n_anchor])
    """
    highq_spots = (((adata_sample.to_df() > 0).sum(axis=1) > np.percentile((adata_sample.to_df() > 0).sum(axis=1), v_low))  &
                   ((adata_sample.to_df()).sum(axis=1) > np.percentile((adata_sample.to_df()).sum(axis=1), v_low))          &
                   ((adata_sample.to_df() > 0).sum(axis=1) < np.percentile((adata_sample.to_df() > 0).sum(axis=1), v_high)) &
                   ((adata_sample.to_df()).sum(axis=1) < np.percentile((adata_sample.to_df()).sum(axis=1), v_high))
                   )

    pure_spots = np.transpose(
        sig_mean.loc[highq_spots, :].index[
            (-np.array(sig_mean.loc[highq_spots, :])).argsort(axis=0)[:n_anchor, :]
        ]
    )
    pure_dict = {
        ct: spot
        for (spot, ct) in zip(pure_spots, sig_mean.columns)
    }

    adata_pure = np.zeros([adata_sample.n_obs, 1])
    adata_pure_idx = [np.where(adata_sample.obs_names == i)[0][0] for i in
                      sorted({x for v in pure_dict.values() for x in v})]
    adata_pure[adata_pure_idx] = 1
    return pure_spots, pure_dict, adata_pure


def get_umap(adata_sample, display=False):
    sc.tl.pca(adata_sample, svd_solver='arpack')
    sc.pp.neighbors(adata_sample, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_sample, min_dist=0.2)
    if display:
        sc.pl.umap(adata_sample)
    umap_plot = pd.DataFrame(adata_sample.obsm['X_umap'],
                             columns=['umap1', 'umap2'],
                             index=adata_sample.obs_names)
    return umap_plot


def get_simu_map_info(umap_plot):
    map_info = []
    map_info = [-umap_plot['umap2'] * 10, umap_plot['umap1'] * 10]
    map_info = pd.DataFrame(np.transpose(map_info),
                            columns=['array_row', 'array_col'],
                            index=umap_plot.index)
    return map_info


def get_windowed_library(adata_sample, map_info, library, window_size):
    library_n = []
    for i in adata_sample.obs_names:
        window_size = window_size
        dist_arr = np.sqrt((map_info.loc[:, 'array_col'] - map_info.loc[i, 'array_col']) ** 2 + (
                map_info.loc[:, 'array_row'] - map_info.loc[i, 'array_row']) ** 2)
        library_n.append(library[dist_arr < window_size].mean())
    library_n = np.array(library_n)
    return library_n


def append_sigs(gene_sig, factor, sigs, n_genes=5):
    """
    Append list of genes to a given cell type as additional signatures or
    add novel cell type / states & their signatures
    """
    assert len(sigs) > 0, "Signature list must have positive length"
    gene_sig_new = gene_sig.copy()
    if not isinstance(sigs, list):
        sigs = sigs.to_list()
    if n_genes < len(sigs):
        sigs = sigs[:n_genes]

    temp = set([i for i in gene_sig[factor] if str(i) != 'nan']+[i for i in sigs if str(i) != 'nan'])
    if len(temp) > gene_sig_new.shape[0]:
        gene_sig_new = gene_sig_new.append(pd.DataFrame([[np.nan]*gene_sig.shape[1]]*(len(temp)-gene_sig_new.shape[0]), columns=gene_sig_new.columns), ignore_index=True)
    else:
        temp = list(temp)+[np.nan]*(gene_sig_new.shape[0]-len(temp))
    gene_sig_new[factor] = list(temp)

    return gene_sig_new


def append_factors(gene_sig, markers_df):
    """
    Append list of archetypes as additional cell type / state (w/ their DEGs as marker genes)
    """
    gene_sig_new = pd.concat((gene_sig, markers_df), axis=1)
    return gene_sig_new


def replace_factors(gene_sig, cols, markers_df, archetypes):
    """
    Replace signatures of given `cols` from original annotation `gene_sig`
    w/ archetypal DEGs in given `archetypes`
    """
    assert (isinstance(cols, str) and isinstance(archetypes, str)) or (len(cols) == len(archetypes)), \
        "Unequal number of cell types to replace from archetypal markers to signatures"
    try:
        gene_sig[cols] = markers_df[archetypes]
    except KeyError:
        LOGGER.info('Invalid cell type / archetype names found, return the original signatures<br>'
                    'please double-check `cols` or `archetypes` values exist in dataframes respectively')
    return gene_sig


def refine_anchors(
        visium_args,
        aa_model,
        thld=0.35,
        n_genes=5,
        n_iters=1
):
    """
    Refine anchor spots & marker genes with archetypal analysis. We append DEGs
    computed from archetypes to their best-matched anchors followed by re-computing
    new anchor spots

    Parameters
    ----------
    visium_args : VisiumArgument
        Default parameter set for Starfysh upon dataloading

    aa_model : ArchetypalAnalysis
        Pre-computed archetype object

    thld : float
        Threshold cutoff for anchor-archetype mapping

    n_genes : int
        # archetypal marker genes to append per refinement iteration

    Returns
    -------
    visimu_args : VisiumArgument
        updated parameter set for Starfysh
    """

    gene_sig = visium_args.gene_sig.copy()
    anchors = visium_args.get_anchors()
    map_df, _ = aa_model.assign_archetypes(anchors) # Retract anchor-archetype mapping scores
    markers_df = aa_model.find_markers(n_markers=50, display=False)

    for iteration in range(n_iters):
        print('Refining round {}...'.format(iteration + 1))
        map_used = map_df.copy()

        # (1). Update signatures
        for i in range(gene_sig.shape[1]):
            selected_arch = map_used.columns[map_used.loc[map_used.index[i],:]>=thld]
            for j in selected_arch:
                print('appending {0} genes in {1} to {2}...'.format(
                    str(n_genes), j, map_used.index[i]
                ))

                gene_sig = append_sigs(
                    gene_sig=gene_sig,
                    factor=map_used.index[i],
                    sigs=markers_df[j],
                    n_genes=n_genes
                )

        # (2). Update anchors & re-compute anchor-archetype mapping
        visium_args.update_anchors(gene_sig)
        anchors = visium_args.get_anchors()
        map_df, _ = aa_model.assign_archetypes(anchors)

    return visium_args
