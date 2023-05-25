import os

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, gaussian_kde


# Module import
from .post_analysis import get_z_umap
from .utils import extract_feature


def plot_spatial_feature(adata_sample,
                         map_info,
                         variable,
                         label
                     ):
    all_loc = np.array(map_info.loc[:,['array_col','array_row']])
    fig,axs= plt.subplots(1,1,figsize=(2.5,2),dpi=300)
    g=axs.scatter(all_loc[:,0],
                  -all_loc[:,1],
                  c=variable,
                  cmap='magma',
                  s=1
                 )
    fig.colorbar(g,label=label)
    plt.axis('off')


def plot_spatial_gene(adata_sample,
                         map_info,
                         gene_name,
                         
                     ):
    all_loc = np.array(map_info.loc[:,['array_col','array_row']])
    fig,axs= plt.subplots(1,1,figsize=(2.5,2),dpi=300)
    g=axs.scatter(all_loc[:,0],
                  -all_loc[:,1],
                  c=adata_sample.to_df().loc[:,gene_name],
                  cmap='magma',
                  s=1
                 )
    fig.colorbar(g,label=gene_name)
    plt.axis('off')


def plot_anchor_spots(umap_plot,
                      pure_spots,
                      sig_mean,
                      bbox_x=2,
                     ):
    fig,ax = plt.subplots(1,1,dpi=300,figsize=(3,3))
    ax.scatter(umap_plot['umap1'],
               umap_plot['umap2'],
               s=2,
               alpha=1,
               color='lightgray')
    for i in range(len(pure_spots)):
        ax.scatter(umap_plot['umap1'][pure_spots[i]],
                   umap_plot['umap2'][pure_spots[i]],
                   s=8)
    plt.legend(['all']+[i for i in sig_mean.columns],
               loc='right', 
               bbox_to_anchor=(bbox_x,0.5),)
    ax.grid(False)
    ax.axis('off')


def plot_evs(evs, kmin):
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(6, 3))
    plt.plot(np.arange(len(evs))+kmin, evs, '.-')
    plt.xlabel('ks')
    plt.ylabel('Explained Variance')
    plt.show()


def pl_spatial_inf_feature(
    adata,
    feature,
    factor=None,
    vmin=0,
    vmax=None,
    spot_size=100,
    alpha=0,
    cmap='Spectral_r'
):
    """Spatial visualization of Starfysh inference features"""
    if isinstance(factor, str):
        assert factor in adata.uns['cell_types'], \
            "Invalid Starfysh inference factor (cell type): ".format(factor)
    elif isinstance(factor, list):
        for f in factor:
            assert f in adata.uns['cell_types'], \
                "Invalid Starfysh inference factor (cell type): ".format(f)
    else:
        factor = adata.uns['cell_types']  # if None, display for all cell types

    adata_pl = extract_feature(adata, feature)

    if feature == 'qc_m':
        if isinstance(factor, list):
            title = [f + ' (Inferred proportion - Spatial)' for f in factor]
        else:
            title = factor + ' (Inferred proportion - Spatial)'
        sc.pl.spatial(
            adata_pl,
            color=factor, spot_size=spot_size, color_map=cmap,
            ncols=3, vmin=vmin, vmax=vmax, alpha_img=alpha,
            title=title, legend_fontsize=8
        )
    elif feature == 'ql_m':
        title = 'Estimated tissue density'
        sc.pl.spatial(
            adata_pl,
            color='density', spot_size=spot_size, color_map=cmap,
            vmin=vmin, vmax=vmax, alpha_img=alpha,
            title=title, legend_fontsize=8
        )
    elif feature == 'qz_m':
        # Visualize deconvolution on UMAP of inferred Z-space
        qz_u = get_z_umap(adata_pl.obs.values)
        qc_df = extract_feature(adata, 'qc_m').obs
        if isinstance(factor, list):
            for cell_type in factor:
                title = cell_type + ' (Inferred proportion - UMAP of Z)'
                pl_umap_feature(qz_u, qc_df[cell_type].values, cmap, title,
                                vmin=vmin, vmax=vmax)
        else:
            title = factor + ' (Inferred proportion - UMAP of Z)'
            pl_umap_feature(qz_u, qc_df[factor].values, cmap, title,
                            vmin=vmin, vmax=vmax)
    else:
        raise ValueError('Invalid Starfysh inference results `{}`, please choose from `qc_m`, `qz_m` & `ql_m`'.format(feature))

    pass


def pl_umap_feature(qz_u, qc, cmap, title, spot_size=3, vmin=0, vmax=None):
    """Single Z-UMAP visualization of Starfysh deconvolutions"""
    fig, axes = plt.subplots(1, 1, figsize=(4, 3), dpi=200)
    g = axes.scatter(
        qz_u[:, 0], qz_u[:, 1],
        cmap=cmap, c=qc, s=spot_size, vmin=vmin, vmax=vmax,
    )
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(title)
    axes.axis('off')

    fig.colorbar(g, label='Inferred proportions')

    pass


def pl_spatial_inf_gene(
    adata,
    factor,
    feature,
    vmin=0,
    vmax=None,
    spot_size=100,
    alpha=0,
    cmap='Spectral_r'
):
    if isinstance(feature, str):
        assert feature in set(adata.var_names), \
            "Gene {0} isn't HVG, please choose from `adata.var_names`".format(feature)
        title = feature + ' (Predicted expression)'
    else:
        for f in feature:
            assert f in set(adata.var_names), \
                "Gene {0} isn't HVG, please choose from `adata.var_names`".format(f)
        title = [f + ' (Predicted expression)' for f in feature]

    # Assign dummy `var_names` to avoid gene name in both obs & var
    adata_expr = extract_feature(adata, factor+'_inferred_exprs')
    adata_expr.var_names = np.arange(adata_expr.shape[1])

    sc.pl.spatial(
        adata_expr,
        color=feature, spot_size=spot_size, color_map=cmap,
        ncols=3, vmin=vmin, vmax=vmax, alpha_img=alpha,
        title=title,
        legend_fontsize=8
    )
    
    pass


# --------------------------
# util funcs for benchmark
# --------------------------

def _dist2gt(A, A_gt):
    """
    Calculate the distance to ground-truth correlation matrix (proportions)
    """
    return np.linalg.norm(A - A_gt, ord='fro')


def bootstrap_dists(corr_df, corr_gt_df, n_iter=1000, size=10):
    """
    Calculate the avg. distance to ground-truth (sub)-matrix based on random subsampling
    """
    if size == None:
        size = corr_df.shape[0]
    n = min(size, size)
    labels = corr_df.columns
    dists = np.zeros(n_iter)

    for i in range(n_iter):
        lbl = np.random.choice(corr_df.columns, n)
        A = corr_df.loc[lbl, lbl].values
        A_gt = corr_gt_df.loc[lbl, lbl].values
        dists[i] = _dist2gt(A, A_gt)

    return dists


def disp_rmse(y_true, y_preds, labels, title=None, return_rmse=False):
    """
    Boxplot of per-spot RMSEs for each prediction
    """
    n_spots, n_cts = y_true.shape
    rmses = np.array([
        np.sqrt(((y_true.values-y_pred.values)**2).sum(1) / n_cts)
        for y_pred in y_preds
    ])

    lbls = np.repeat(labels, n_spots)
    df = pd.DataFrame({
        'RMSE': rmses.flatten(),
        'Method': lbls
    })
    plt.figure(figsize=(10, 6))
    g = sns.boxplot(x='Method', y='RMSE', data=df)
    g.set_xticklabels(labels, rotation=60)
    plt.suptitle(title)
    plt.show()

    return rmses if return_rmse else None


def disp_corr(
    y_true, y_pred,
    outdir=None,
    figsize=(3.2, 3.2),
    fontsize=5,
    title=None,
    filename=None,
    savefig=False,
    format='png',
    return_corr=False
):
    """
    Calculate & plot correlation of cell proportion (or absolute cell abundance)
    between ground-truth & predictions (both [S x F])
    """

    assert y_true.shape[0] == y_pred.shape[0], 'Inconsistent sample sizes between ground-truth & prediction'
    if savefig:
        assert format == 'png' or format == 'eps' or format == 'svg', "Invalid saving format"

    v1 = y_true.values
    v2 = y_pred.values

    n_factor1, n_factor2 = v1.shape[1], v2.shape[1]
    corr = np.zeros((n_factor1, n_factor2))
    gt_corr = y_true.corr().values

    for i in range(n_factor1):
        for j in range(n_factor2):
            corr[i, j], _ = np.round(pearsonr(v1[:, i], v2[:, j]), 3)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax = sns.heatmap(
        corr, annot=True,
        cmap='RdBu_r', vmin=-1, vmax=1,
        annot_kws={"fontsize": fontsize},
        cbar_kws={'label': 'Cell type proportion corr.'},
        ax=ax
        )

    ax.set_xticks(np.arange(n_factor2) + 0.5)
    ax.set_yticks(np.arange(n_factor1) + 0.5)
    ax.set_xticklabels(y_pred.columns, rotation=90)
    ax.set_yticklabels(y_true.columns, rotation=0)
    ax.set_xlabel('Estimated proportion')
    ax.set_ylabel('Ground truth proportion')

    if title is not None:
        # ax.set_title(title+'\n'+'Distance = %.3f' % (dist2identity(corr)))
        ax.set_title(title + '\n' + 'Distance = %.3f' % (dist2gt(corr, gt_corr)))

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    if savefig and (outdir is not None and filename is not None):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig(os.path.join(outdir, filename + '.' + format), bbox_inches='tight', format=format)
    plt.show()

    return corr if return_corr else None


def disp_prop_scatter(
    y_true, y_pred,
    outdir=None,
    filename=None,
    savefig=False,
    format='png'
):
    """
    Scatter plot of spot-wise proportion between ground-truth & predictions
    """
    assert y_true.shape == y_pred.shape, 'Inconsistent dimension between ground-truth & prediction'
    if savefig:
        assert format == 'png' or format == 'eps' or format == 'svg', "Invalid saving format"

    n_factors = y_true.shape[1]
    y_true_vals = y_true.values
    y_pred_vals = y_pred.values
    ncols = int(np.ceil(n_factors / 2))

    fig, (ax1, ax2) = plt.subplots(2, ncols, figsize=(2 * ncols, 4.4), dpi=300)

    for i in range(n_factors):
        v1 = y_true_vals[:, i]
        v2 = y_pred_vals[:, i]
        r2 = r2_score(v1, v2)

        v_stacked = np.vstack([v1, v2])
        den = gaussian_kde(v_stacked)(v_stacked)

        ax = ax1[i] if i < ncols else ax2[i % ncols]
        ax.scatter(v1, v2, c=den, s=.2, cmap='turbo', vmax=den.max() / 3)

        ax.set_aspect('equal')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axis('equal')

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_title(y_pred.columns[i])
        ax.annotate(r"$R^2$ = {:.3f}".format(r2), (0, 1), fontsize=8)

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks(np.arange(0, 1.1, 0.5))
        ax.set_yticks(np.arange(0, 1.1, 0.5))

        ax.set_xlabel('Ground truth proportions')
        ax.set_ylabel('Predicted proportions')

    plt.tight_layout()
    if savefig and (outdir is not None and filename is not None):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig(os.path.join(outdir, filename + '.' + format), bbox_inches='tight', format=format)

    plt.show()

