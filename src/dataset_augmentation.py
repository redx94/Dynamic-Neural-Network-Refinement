
import torch
import torch.nn as nn

class ConditionalGAN(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def generate(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.generator(x)
        
    def discriminate(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.discriminator(x)
