Installation
************
Currently we support local installation from Starfysh root directory:

.. code-block:: bash

    # install
    python setup.py install --user

    # uninstall
    pip uninstall starfysh

Quickstart
**********

.. code-block:: python

    import numpy as np
    import pandas as pd
    import torch
    from starfysh import (
        AA,
        dataloader,
        starfysh,
        utils,
        plot_utils,
        post_analysis
    )

    # (1) Loading dataset & signature gene sets
    data_path = 'data/' # specify data directory
    sig_path = 'signature/signatures.csv' # specify signature directory
    sample_id = 'CID44971_TNBC'

    # --- (a) ST matrix ---
    adata, adata_norm = utils.load_adata(
        data_path,
        sample_id,
        n_genes=2000
    )

    # --- (b) paired H&E image + spots info ---
    hist_img, map_info = utils.preprocess_img(
        data_path,
        sample_id,
        adata.obs.index,
        hchannal=False
    )

    # --- (c) signature gene sets ---
    gene_sig = utils.filter_gene_sig(
        pd.read_csv(sig_path),
        adata.to_df()
    )

    # (2) Starfysh deconvolution

    # --- (a) Preparing arguments for model training
    args = utils.VisiumArguments(
        adata,
        adata_norm,
        gene_sig,
        map_info,
        n_anchors=60, # number of anchor spots per cell-type
        window_size=5  # library size smoothing radius
    )

    adata, adata_noprm = args.get_adata()

    # --- (b) Model training ---
    n_restarts = 3
    epochs = 100
    patience = 10
    device = torch.device('cpu')

    model, loss = utils.run_starfysh(
        args,
        n_restarts,
        epochs=epochs,
        patience=patience
    )

    # (3). Parse deconvolution outputs
    inferences, generatives, px = starfysh.model_eval(
        model,
        adata,
        args.sig_mean,
        device,
        args.log_lib,
    )

