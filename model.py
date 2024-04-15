#Derived from https://github.com/QingyuZhao/VAE-for-Regression
#https://arxiv.org/abs/1904.05948

import torch
from torch import nn
from collections import OrderedDict
import numpy as np
from pytorch_msssim import ssim


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("LeakyReLU_%d" % i, nn.LeakyReLU(negative_slope=0.01, inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, npred = 0):
        #nhid is latent space dimension
        #npred is property dimension
        #shape is image shape
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        self.pool1 = nn.MaxPool2d(2,2, return_indices=True)
        self.pool2 = nn.MaxPool2d(2,2, return_indices=True)
        self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0, bias = False), nn.BatchNorm2d(16), nn.LeakyReLU(negative_slope=0.001, inplace=True), #16
                                    nn.Conv2d(16, 32, 5, padding = 0, bias = False), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.001, inplace=True), #32
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding = 0, bias = False), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.001, inplace=True), #64
                                    nn.Conv2d(64, 64, 3, padding = 0, bias = False), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.001, inplace=True), #64
                                    nn.MaxPool2d(2, 2),
                                    Flatten(), MLP([ww*hh*64, 64, 512-npred]) #64
                                   )
                                   
        self.calc_mean = MLP([512, 256, nhid], last_activation = False) #Reparameterization trick
        self.calc_logvar = MLP([512, 256, nhid], last_activation = False) #Reparameterization trick

    def forward(self, x, pred = None):
        #x is image
        #pred is predicted property
        x = self.encode(x)
        if pred is not None:
            x = torch.cat((x, pred), axis = 1)
        return self.calc_mean(x), self.calc_logvar(x)

        
class Regressor(nn.Module):
    def __init__(self, shape, npred):
        #nhid is latent space dimension
        #npred is property dimension
        #shape is image shape
        super(Regressor, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        self.predict = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0, bias = False), nn.BatchNorm2d(16), nn.LeakyReLU(negative_slope=0.01, inplace=True), #16
                                    nn.Conv2d(16, 32, 5, padding = 0, bias = False), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01, inplace=True), #32
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding = 0, bias = False), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01, inplace=True), #64
                                    nn.Conv2d(64, 64, 3, padding = 0, bias = False), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01, inplace=True), #64
                                    nn.MaxPool2d(2, 2),
                                    Flatten(), MLP([ww*hh*64, 64, 256])
                                   )
        self.calc_mean = MLP([256, 16, npred, 1], last_activation = False)
        self.calc_logvar = MLP([256, 16, npred, 1], last_activation = False)
    def forward(self, x):
        #X is the image
        #predicts property from image
        x = self.predict(x)
        return self.calc_mean(x), self.calc_logvar(x) #reparameterization trick  

class Decoder(nn.Module):
    def __init__(self, shape, nhid=16, npred=1):
        super(Decoder, self).__init__()
        c, w, h = shape

        # Adjust the MLP layers to include deconvolutions
        self.decoder = nn.Sequential(
            # Start with dense layers
            nn.Linear(nhid + npred, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            
            nn.Linear(2048, 512 * 4 * 4),  # Assuming we want to start deconvolutions with 4x4 feature maps
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512 * 4 * 4),

            # Reshape for deconvolutions
            Reshape((-1, 512, 4, 4)),

            # Begin deconvolution layers
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias = False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias = False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias = False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias = False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.BatchNorm2d(32),

            # nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias = False),
            # nn.LeakyReLU(inplace=True, negative_slope=0.01),
            # nn.BatchNorm2d(16),

            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1, bias = False),
            nn.Sigmoid()  # To produce pixel values between [0, 1]
        )

    def forward(self, z, r=None):
        if r is not None:
            z = torch.cat((z, r), dim=1)
        
        # Pass through the decoder
        return self.decoder(z)

# Helper layer for reshaping within Sequential
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class VAE_Regression(nn.Module):
    def __init__(self, shape, nhid = 64, npred = 1, device = 'cpu'):
        #nhid is latent space dimension
        #npred is property dimension
        #shape is image shape
        super(VAE_Regression, self).__init__()
        self.dim = nhid + npred
        self.nhid = nhid
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid, npred)
        self.regressor = Regressor(shape, npred)
        self.device = device
        self.npred= npred
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    def forward(self, x):
        mean_z, logvar_z = self.encoder(x)
        z = self.sampling(mean_z, logvar_z)
        mean_r, logvar_r = self.regressor(x)
        r = self.sampling(mean_r, logvar_r)
        return self.decoder(z,r), mean_z, logvar_z, mean_r, logvar_r, z
    
def loss_regression(X, X_hat, z, z_mean, z_logvar, r, r_mean, r_logvar):
    # Assuming X_hat is the output of a sigmoid activation function
    # BCE Loss for reconstruction
    reconstruction_loss = nn.BCELoss(reduction='sum')(X_hat, X) / X.size(0)

    # KL Divergence (unchanged)
    KL_divergence = (-0.5 * torch.sum(1 + z_logvar - torch.exp(z_logvar) - (z_mean - z)**2)) / 5000

    # Label loss (unchanged)
    label_loss = torch.mean(0.5 * ((((r_mean.squeeze(1) - r)**2) / torch.exp(r_logvar)) + r_logvar)) * 1000000

    # Combined loss
    total_loss = reconstruction_loss + KL_divergence + label_loss
    return total_loss, reconstruction_loss, label_loss, KL_divergence

class VAE_Classic(nn.Module):
    def __init__(self, shape, nhid = 64, npred = 1, device = 'cpu'):
        #nhid is latent space dimension
        #npred is property dimension
        #shape is image shape
        super(VAE_Regression, self).__init__()
        self.dim = nhid + npred
        self.nhid = nhid
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid, npred)
        # self.regressor = Regressor(shape, npred)
        self.device = device
        self.npred= npred
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    def forward(self, x):
        mean_z, logvar_z = self.encoder(x)
        z = self.sampling(mean_z, logvar_z)
        # mean_r, logvar_r = self.regressor(x)
        # r = self.sampling(mean_r, logvar_r)
        return self.decoder(z, None), mean_z, logvar_z, None, None, z
    
def loss_classic(X, X_hat, z, z_mean, z_logvar):
    #This is the loss function from the paper. The problem is that each term is not at the same scale so different terms essentially have different importance
    reconstruction_loss = nn.MSELoss()(X, X_hat)*2
    KL_divergence = (-0.5 * torch.sum(1 + z_logvar - torch.exp(z_logvar) - (z_mean - z)**2))/5000
    # label_loss = torch.mean(0.5*((((r_mean.squeeze(1)-r)**2)/torch.exp(r_logvar))+r_logvar))*100000
    # label_loss = nn.MSELoss()(r_mean.squeeze(1).detach(), r)*10999
    # print(f'reconstruction_loss = {reconstruction_loss}, kl_divergence = {KL_divergence}, label_loss = {label_loss}, mean_loss = {torch.mean(reconstruction_loss+KL_divergence+label_loss)}')
    return torch.mean(reconstruction_loss+KL_divergence), reconstruction_loss, 0, KL_divergence
