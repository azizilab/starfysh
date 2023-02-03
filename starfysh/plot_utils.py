import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


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
