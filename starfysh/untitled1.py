from __future__ import print_function

import numpy as np
import pandas as pd
import os
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.utils import make_grid
from torch.distributions import constraints, Distribution, Normal, Gamma, Poisson, Dirichlet
from torch.distributions import kl_divergence as kl

# Module import
from starfysh import LOGGER

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


# TODO:
#  inherit `AVAE` (expr model) w/ `AVAE_PoE` (expr + histology model), update latest PoE model
class AVAE(nn.Module):
    """ 
    Model design
        p(x|z)=f(z)
        p(z|x)~N(0,1)
        q(z|x)~g(x)
    """
    
    def __init__(
        self,
        adata,
        gene_sig,
        win_loglib,
        alpha_mul=50,
        batch_size=32,

        # DEBUG: whether to regularize non-anchors?
        reg_nonanchors=True,
        test_prior = 0.2,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """
        Auxiliary Variational AutoEncoder (AVAE) - Core model for
        spatial deconvolution without H&E image integration

        Paramters
        ---------
        adata : sc.AnnData
            ST raw expression count (dim: [S, G])

        gene_sig : pd.DataFrame
            Normalized avg. signature expressions for each annotated cell type

        win_loglib : float
            Log-library size smoothed with neighboring spots

        alpha_mul : float (default=1e3)
            Multiplier of Dirichlet concentration parameter to control
            signature prior's confidence
        """
        super().__init__()
        self.win_loglib=torch.Tensor(win_loglib)

        self.c_in = adata.shape[1]  # c_in : Num. input features (# input genes)
        self.c_bn = 10  # c_bn : latent number, numbers of bottle-necks
        self.c_hidden = 256
        self.c_kn = gene_sig.shape[1]
        self.eps = 1e-5  # for r.v. w/ numerical constraints
        self.device = device
        
        self.alpha = torch.ones(self.c_kn)*alpha_mul
        self.alpha = self.alpha.to(device)

        # DEBUG: set up non-informative, sparse Dirichlet prior for non-anchors
        # test whether to regularize non-anchors
        self.reg_na = reg_nonanchors
        self.pc_na = torch.ones(batch_size, self.c_kn) * test_prior # test_prior = 0.2
        self.pc_na = self.pc_na.to(device)
        
        self.qs_logm = torch.nn.Parameter(torch.zeros(self.c_kn, self.c_bn), requires_grad=True)
        self.qu_m = torch.nn.Parameter(torch.randn(self.c_kn, self.c_bn), requires_grad=True)
        self.qu_logv = torch.nn.Parameter(torch.zeros(self.c_kn, self.c_bn), requires_grad=True)

        self.c_enc = nn.Sequential(
                                nn.Linear(self.c_in, self.c_hidden, bias=True),
                                nn.BatchNorm1d(self.c_hidden, momentum=0.01,eps=0.001),
                                nn.ReLU()
        )
        
        self.c_enc_m = nn.Sequential(
                                nn.Linear(self.c_hidden, self.c_kn, bias=True),
                                nn.BatchNorm1d(self.c_kn, momentum=0.01,eps=0.001),
                                nn.Softmax(dim=-1)
        )
        
        self.l_enc = nn.Sequential(
                                nn.Linear(self.c_in, self.c_hidden, bias=True),
                                nn.BatchNorm1d(self.c_hidden, momentum=0.01,eps=0.001),
                                nn.ReLU(),
                                #nn.Linear(self.c_hidden, 1, bias=True),
                                #nn.ReLU(),
        )
        
        self.l_enc_m = nn.Linear(self.c_hidden, 1)
        self.l_enc_logv = nn.Linear(self.c_hidden, 1)
        
        # neural network f1 to get the z, p(z|x), f1(x,\phi_1)=[z_m,torch.exp(z_logv)]
        self.z_enc = nn.Sequential(
                                #nn.Linear(self.c_in+self.c_kn, self.c_hidden, bias=True),
                                nn.Linear(self.c_in, self.c_hidden, bias=True),
                                nn.BatchNorm1d(self.c_hidden, momentum=0.01,eps=0.001),
                                nn.ReLU(),
        )
        
        self.z_enc_m = nn.Linear(self.c_hidden, self.c_bn *  self.c_kn)
        self.z_enc_logv = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)
        
        # gene dispersion
        self._px_r = torch.nn.Parameter(torch.randn(self.c_in),requires_grad=True)

        # neural network g to get the x_m and x_v, p(x|z), g(z,\phi_3)=[x_m,x_v]
        self.px_hidden_decoder = nn.Sequential(
                                nn.Linear(self.c_bn, self.c_hidden, bias=True),
                                nn.ReLU(),         
        )
        self.px_scale_decoder = nn.Sequential(
                              nn.Linear(self.c_hidden,self.c_in),
                              nn.Softmax(dim=-1)
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample
    
    def inference(self, x):
        x_n = torch.log1p(x)
        hidden = self.l_enc(x_n)
        ql_m = self.l_enc_m(hidden)
        ql_logv = self.l_enc_logv(hidden)
        ql = self.reparameterize(ql_m, ql_logv)

        x_n = torch.log1p(x)
        hidden = self.c_enc(x_n)
        qc_m = self.c_enc_m(hidden)
        qc = Dirichlet(qc_m * self.alpha + self.eps).rsample()[:,:,None]
        hidden = self.z_enc(x_n)

        qz_m_ct = self.z_enc_m(hidden).reshape([x_n.shape[0],self.c_kn,self.c_bn])
        qz_m_ct = (qc * qz_m_ct)
        qz_m = qz_m_ct.sum(axis=1)

        qz_logv_ct = self.z_enc_logv(hidden).reshape([x_n.shape[0],self.c_kn,self.c_bn])
        qz_logv_ct = (qc * qz_logv_ct)
        qz_logv = qz_logv_ct.sum(axis=1)
        qz = self.reparameterize(qz_m, qz_logv)

        qu_m = self.qu_m
        qu_logv = self.qu_logv
        qu = self.reparameterize(qu_m, qu_logv)

        return dict(
                    qc_m = qc_m,
                    qc=qc,
                    qz_m=qz_m,
                    qz_m_ct=qz_m_ct,
                    qz_logv = qz_logv,
                    qz_logv_ct = qz_logv_ct,
                    qz=qz,
                    ql_m=ql_m,
                    ql_logv=ql_logv,
                    ql=ql,
                    qu_m=qu_m,
                    qu_logv=qu_logv,
                    qu=qu,
                    qs_logm=self.qs_logm,
                   )
    
    def generative(
        self,
        inference_outputs,
        xs_k,
        anchor_idx
    ):
        
        qz = inference_outputs['qz']
        ql = inference_outputs['ql']

        hidden = self.px_hidden_decoder(qz)
        px_scale = self.px_scale_decoder(hidden)
        self.px_rate = torch.exp(ql) * px_scale + self.eps
        pc_p = xs_k + self.eps

        return dict(
            px_rate=self.px_rate,
            px_r=self.px_r,
            pc_p=pc_p,
            xs_k=xs_k,
        )
    
    def get_loss(
        self,
        generative_outputs,
        inference_outputs,
        x,
        x_peri,
        library,
        device
    ):
    
        qc = inference_outputs["qc"]
        qc_m = inference_outputs["qc_m"]

        qs_logm = self.qs_logm
        qu = inference_outputs["qu"]
        qu_m = inference_outputs["qu_m"]
        qu_logv = inference_outputs["qu_logv"]

        qz_m = inference_outputs["qz_m"]
        qz_logv = inference_outputs["qz_logv"]

        ql_m = inference_outputs["ql_m"]
        ql_logv = inference_outputs['ql_logv']
        ql = inference_outputs['ql']
        
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        pc_p = generative_outputs["pc_p"]


        # Regularization terms
        pu_m = torch.zeros_like(qu_m)
        pu_std = torch.ones_like(qu_logv) * 10
        kl_divergence_u = kl(
            Normal(qu_m, torch.exp(qu_logv / 2)),
            Normal(pu_m, pu_std)
        ).sum(dim=1).mean()

        mean_pz = (qu.unsqueeze(0) * qc).sum(axis=1)
        std_pz = (torch.exp(qs_logm / 2).unsqueeze(0) * qc).sum(axis=1)
        kl_divergence_z = kl(
            Normal(qz_m, torch.exp(qz_logv / 2)),
            Normal(mean_pz, std_pz)
        ).sum(dim=1).mean()

        kl_divergence_n = kl(
            Normal(ql_m, torch.exp(ql_logv / 2)),
            Normal(library, torch.ones_like(ql))
        ).sum(dim=1).mean()

        # DEBUG: test what if we set uninformative but sparse uniform Dirichlet for non-anchors
        kl_divergence_c = torch.tensor([0.0]).to(self.device)
        anchor_indices = x_peri[:, 0] == 1
        na_indices = x_peri[:, 0] == 0
        if anchor_indices.sum() > 0:
            kl_divergence_c += kl(
                Dirichlet(qc_m[anchor_indices] * self.alpha),
                Dirichlet(pc_p[anchor_indices] * self.alpha)
            ).mean()
        if na_indices.sum() > 0 and self.reg_na:
            pc_na = self.pc_na[:na_indices.shape[0]]  # edge condition: last batch w/ shape < batch-size
            kl_divergence_c += kl(
                Dirichlet(qc_m[na_indices] * self.alpha),
                Dirichlet(pc_na[na_indices])
            ).mean()


        # Reconstruction term
        reconst_loss = -NegBinom(px_rate, torch.exp(px_r)).log_prob(x).sum(-1).mean()
        
        reconst_loss = reconst_loss.to(device)
        kl_divergence_u = kl_divergence_u.to(device)
        kl_divergence_z = kl_divergence_z.to(device)
        kl_divergence_c = kl_divergence_c.to(device)
        kl_divergence_n = kl_divergence_n.to(device)
        loss = reconst_loss + kl_divergence_u + kl_divergence_z + kl_divergence_c + kl_divergence_n 

        return (loss,
                reconst_loss,
                kl_divergence_u,
                kl_divergence_z,
                kl_divergence_c,
                kl_divergence_n
               )
    
    @property
    def px_r(self):
        return F.softplus(self._px_r) + self.eps
