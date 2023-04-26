import torch
import torch.nn as nn

class ConvLayer(nn.Module):

    def __init__(self, 
                n_in,
                n_out,
                hidden_dims,
                label_dim,
                 
                encoder = True,                    
                ):
        super(ConvLayer, self).__init__()
        self.encoder = encoder
        modules = []
        if encoder:
            n_in = n_in + label_dim 
        else:
            n_in = n_in + label_dim
        
        
        for i in range(len(hidden_dims)):
            if encoder:
                conv_l= nn.Conv2d(n_in,
                            hidden_dims[i],
                            kernel_size=3,
                            stride = 1,
                            padding=1,
                            )
            else:
                conv_l= nn.ConvTranspose2d(n_in,
                            hidden_dims[i],
                            kernel_size=3,
                            stride = 1,
                            padding=1,
                            output_padding=0)
            modules.append(
                nn.Sequential(
                    conv_l,
                    nn.BatchNorm2d(hidden_dims[i]),
                    nn.LeakyReLU())
            )
            n_in = hidden_dims[i]+label_dim

        self.layers = nn.Sequential(*modules)
    
    def forward(self,x,y,label):
        
        
        if self.encoder:
            x = x.transpose(2,3).transpose(1,2)
            y = y.transpose(2,3).transpose(1,2)
            input = torch.cat([x,y],dim=1)
        else:
            input = x
        
        label = label.transpose(2,3).transpose(1,2)
        for i, layers in enumerate(self.layers):
            for layer in layers:
                if (isinstance(layer, nn.BatchNorm2d))|(isinstance(layer, nn.LeakyReLU)):
                    
                    input = layer(input)
                else:
                    
                    
                    input = torch.cat([input,label],dim=1)
                    
                    input = layer(input)
        return input