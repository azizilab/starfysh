import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from typing import List,Dict, Optional, OrderedDict, Sequence, Union
from torch.autograd import Variable
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
from . import ConvLayer

def reparameterize_gaussian(mu,var):
    return Normal(mu, var.sqrt()).rsample()
class ConvEncoder(nn.Module):
    def __init__(self, 
                rna_dim: int = 317,
                label_dim: int = 15,
                #img_dim: int =3,
                hidden_dims: List = [128,64],
                latent_dim: int = 20,
                ):
        super(ConvEncoder, self).__init__()
        #self.input_dim=input_dim
        #self.output_dim=output_dim

        ## z_encoder for latent
        self.z_encoder = ConvLayer(
                            n_in = rna_dim,
                            n_out = hidden_dims[-1],
                            hidden_dims = hidden_dims,
                            label_dim = label_dim, 
                            encoder = True)
        
        n_in = hidden_dims[-1]
        self.z_mean = nn.Sequential(
                                       nn.Conv2d(n_in, out_channels=latent_dim, kernel_size= 3, stride= 1, padding  = 1),
                                       nn.BatchNorm2d(latent_dim),
                                       #nn.Dropout(p=0.1),
                                       nn.LeakyReLU()  
        )   
        self.z_var = nn.Sequential(
                                       nn.Conv2d(n_in, out_channels=latent_dim, kernel_size= 3, stride= 1, padding  = 1),
                                       nn.BatchNorm2d(latent_dim),
                                       #nn.Dropout(p=0.1),
                                       nn.LeakyReLU()  
        )   

    
    def forward(self, x,gamma,label):
        var_eps = 1e-4
        q_ = self.z_encoder(x, gamma,label)
        
        qz_mean = self.z_mean(q_)
        qz_var = torch.exp(0.5*self.z_var(q_)) + var_eps ## 0.5 maybe not necessary
        z_latent = reparameterize_gaussian(qz_mean, qz_var)

        ql = self.l_encoder(x,y,label)
        ql_mean = self.l_mean(ql)
        ql_var = torch.exp(0.5*self.l_var(ql)) 
        libary = torch.exp(torch.clamp(reparameterize_gaussian(ql_mean, ql_var),max=20))
    
        q_gamma = self.gamma_encoder(x,y,label)
        qgamma_mean = self.gamma_mean(q_gamma)
        qgamma_var = torch.exp(0.5*self.gamma_var(q_gamma))  
        gamma_density = F.softmax(torch.exp(reparameterize_gaussian(qgamma_mean, qgamma_var)),dim=-1)
        
        infer_output = {}
        infer_output['qz_mean']=qz_mean
        infer_output['qz_var']=qz_var
        infer_output['z_latent']=z_latent
        infer_output['ql_mean']=ql_mean
        infer_output['ql_var']=ql_var
        infer_output['libary']=libary
        infer_output['qgamma_mean']=qgamma_mean
        infer_output['qgamma_var']=qgamma_var
        infer_output['gamma_density']=gamma_density
        
        return infer_output
    
    
class ConvDecoder(nn.Module):
    def __init__(self, 
                latent_dim: int = 20,
                hidden_dims =  [128,64],
                output_dim: int = 317,
                label_dim: int = 15, 
                img_dim: int =3,                
                ):
        super(ConvDecoder, self).__init__()


        hidden_dims.reverse()
        
       
        self.z_decoder = ConvLayer(
                            n_in = latent_dim,
                            n_out = hidden_dims[-1],
                            hidden_dims = hidden_dims,
                            label_dim = label_dim,
                            encoder = False)

    
        self.px_decoder = nn.Sequential(                            
                            nn.ConvTranspose2d(hidden_dims[-1], 
                                               out_channels= output_dim,
                                               kernel_size= 3,
                                               padding= 1),
                            nn.Softplus()
                            )
        

        
    def forward(self, z, y, library,label):
         
        h = self.z_decoder(z, y,label)
        px = self.px_decoder(h)
        px_rate = library*nn.Softmax(dim=-1)(px)
        #hi = self.gamma_decoder(z,y,label)
        #pi = self.pi_decoder(hi)
        #pi_rate = library*nn.Softmax(dim=-1)(pi)
        
        genera_output = {}
        genera_output['px_rate']=px_rate
        #genera_output['pi_rate']=pi_rate

        return genera_output