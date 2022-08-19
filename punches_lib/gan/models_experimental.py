import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, base_width=64, num_channels=512):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.base_width = base_width
        self.num_channels = num_channels
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d( self.latent_dim, self.base_width * 8, 1, 1, 1, bias=False),
            nn.BatchNorm2d(self.base_width * 8),
            nn.ReLU(True),
            # state size. (self.base_width*8) x 4 x 4
            nn.Conv2d(self.base_width * 8, self.base_width * 4, 1, 2, 0, bias=False),
            nn.BatchNorm2d(self.base_width * 4),
            nn.ReLU(True),
            # state size. (self.base_width*4) x 8 x 8
            nn.Conv2d( self.base_width * 4, self.base_width * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width * 2),
            nn.ReLU(True),
            # state size. (self.base_width*2) x 16 x 16
            nn.Conv2d( self.base_width * 2, self.base_width*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width*4),
            nn.ReLU(True),
            # state size. (self.base_width) x 32 x 32
            nn.Conv2d( self.base_width*4, self.num_channels, 1, 1, 0, bias=True),
            #nn.Tanh()
            # state size. (self.num_channels) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, num_channels=512, base_width=64):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.base_width = base_width
        self.main = nn.Sequential(
            nn.Conv2d(self.num_channels, self.base_width*8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width*8, self.base_width*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width*4, self.base_width*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width*2, self.base_width, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)