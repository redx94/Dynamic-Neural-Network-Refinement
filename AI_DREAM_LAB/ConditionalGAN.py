
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, condition):
        x_cond = torch.cat([x, condition], dim=1)
        return self.net(x_cond)

class ConditionalGAN:
    def __init__(self, latent_dim, condition_dim, output_dim, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, condition_dim, output_dim).to(device)
        self.discriminator = Discriminator(output_dim, condition_dim).to(device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        
    def train_step(self, real_samples, conditions):
        batch_size = real_samples.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_samples = self.generator(z, conditions)
        
        d_loss_real = self.criterion(self.discriminator(real_samples, conditions), real_labels)
        d_loss_fake = self.criterion(self.discriminator(fake_samples.detach(), conditions), fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        g_loss = self.criterion(self.discriminator(fake_samples, conditions), real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item()
        }
    
    def generate(self, conditions, num_samples=1):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z, conditions)
        return samples
