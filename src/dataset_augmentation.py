
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)

class ConditionalGAN:
    def __init__(self, latent_dim, condition_dim, output_dim):
        self.generator = Generator(latent_dim, condition_dim, output_dim)
        self.discriminator = Discriminator(output_dim, condition_dim)
        self.latent_dim = latent_dim
        
    def generate_samples(self, conditions, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        with torch.no_grad():
            return self.generator(z, conditions)
            
    def train_step(self, real_samples, conditions, optimizer_g, optimizer_d):
        batch_size = real_samples.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        optimizer_d.zero_grad()
        z = torch.randn(batch_size, self.latent_dim)
        fake_samples = self.generator(z, conditions)
        
        d_real = self.discriminator(real_samples, conditions)
        d_fake = self.discriminator(fake_samples.detach(), conditions)
        
        d_loss = (nn.BCELoss()(d_real, real_labels) + 
                 nn.BCELoss()(d_fake, fake_labels)) / 2
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        d_fake = self.discriminator(fake_samples, conditions)
        g_loss = nn.BCELoss()(d_fake, real_labels)
        g_loss.backward()
        optimizer_g.step()
        
        return d_loss.item(), g_loss.item()
