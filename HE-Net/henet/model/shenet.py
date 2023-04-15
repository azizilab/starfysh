import torch
import torch.nn as nn
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
from . import ConvEncoder,ConvDecoder


class sheNET(nn.Module):
    def __init__(self, 
                 ):
        super(sheNET, self).__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def generative(self, z, y,libary,label_stack):
        
        genera_output = self.decoder(z,y,libary,label_stack)
        return genera_output
    
    def inference(self,x,y,label):

        x_ = torch.log(1+x)
        infer_output  = self.encoder(x_,gamma,label)
        
        return infer_output

    def loss(self, x, y, infer_output,genera_output):
        """reconstruction+KLD"""
        var_eps = 1e-4
        qz_m = infer_output["qz_mean"]
        qz_v = infer_output["qz_var"]
        ql_m = infer_output["ql_mean"]
        ql_v = infer_output["ql_var"]
        qg_m = infer_output['qgamma_mean']
        qg_v = infer_output['qgamma_var']
        px_ = genera_output["px_rate"]
        py_ = genera_output["pi_rate"]
        
        
        px_scale = torch.exp(0.5*torch.nn.Parameter(torch.randn(px_.shape)))+var_eps
        py_scale = torch.exp(0.5*torch.nn.Parameter(torch.randn(py_.shape)))+var_eps
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        px_scale = px_scale.to(device)
        py_scale = py_scale.to(device)
        x = x.transpose(2,3).transpose(1,2)
        y = y.transpose(2,3).transpose(1,2)
        ## get the reconstruction loss
        print(px_.shape)
        print(x.shape)
        #recon_loss_rna = nn.MSELoss(px_,x)
        #recon_loss_img = nn.MSELoss(py_,y)
        
        recon_loss_rna = -Normal(px_, torch.sqrt(torch.Tensor(1))).log_prob(x).mean(-1).mean(-1).mean(-1)
        recon_loss_img = -Normal(py_, torch.sqrt(torch.Tensor(1))).log_prob(y).mean(-1).mean(-1).mean(-1)

        ## get the KL divergence
        kld_z = kl(Normal(qz_m,torch.sqrt(qz_v)), Normal(0,1)).mean(dim=1).mean(dim=1).mean(dim=1)
        kld_l = kl(Normal(ql_m,torch.sqrt(ql_v)), Normal(0,1)).mean(dim=1).mean(dim=1).mean(dim=1)
        kld_gamma = kl(Normal(qg_m,torch.sqrt(qg_v)), Normal(0,1)).mean(dim=1).mean(dim=1).mean(dim=1)

        #loss = torch.mean(recon_loss_rna+recon_loss_img+kld_z+kld_l+kld_gamma)
        loss = torch.mean(recon_loss_rna)

        return loss

