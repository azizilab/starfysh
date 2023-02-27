Installation
************
.. code-block:: bash

    pip install Starfysh

Quickstart
**********

.. code-block:: python

    import os
    import numpy as np
    import pandas as pd
    import torch

    from starfysh import (AA, utils, plot_utils, post_analysis)
    from starfysh import starfysh as sf_model

    # (1) Loading dataset & signature gene sets
    data_path = 'data/' # specify data directory
    sig_path = 'signature/signatures.csv' # specify signature directory
    sample_id = 'SAMPLE_ID'

    # --- (a) ST matrix ---
    adata, adata_norm = utils.load_adata(
        data_path,
        sample_id,
        n_genes=2000
    )

    # --- (b) paired H&E image + spots info ---
    img_metadata = utils.preprocess_img(
        data_path,
        sample_id,
        adata_index=adata.obs.index,
        hchannel=False
    )

    # --- (c) signature gene sets ---
    gene_sig = utils.filter_gene_sig(
        pd.read_csv(sig_path),
        adata.to_df()
    )

    # (2) Starfysh deconvolution

    # --- (a) Preparing arguments for model training
    args = utils.VisiumArguments(adata,
                                 adata_normed,
                                 gene_sig,
                                 img_metadata,
                                 n_anchors=60,
                                 window_size=3,
                                 sample_id=sample_id
                                 )

    adata, adata_normed = args.get_adata()
    anchors_df = args.get_anchors()

    # --- (b) Model training ---
    n_restarts = 3
    epochs = 200
    patience = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run models
    model, loss = utils.run_starfysh(
        visium_args,
        n_repeats=n_repeats,
        epochs=epochs,
        patience=patience,
        device=device
    )

    # (3). Parse deconvolution outputs
    inference_outputs, generative_outputs = sf_model.model_eval(
        model,
        adata,
        visium_args,
        poe=False,
        device=device
    )

