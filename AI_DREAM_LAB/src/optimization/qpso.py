import numpy as np
import torch

class QPSOOptimizer:
    def __init__(self, model, n_particles, max_iter, omega=0.8, phi_p=2.05, phi_g=2.05):
        self.model = model
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.omega = omega  # Inertia weight
        self.phi_p = phi_p  # Cognitive coefficient
        self.phi_g = phi_g  # Social coefficient
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.best_position = None
        self.particles = []
        self.pbest_positions = []
        self.pbest_losses = []
        self._initialize_particles()

    def _initialize_particles(self):
        """Initialize particles with random positions and velocities."""
        for _ in range(self.n_particles):
            # Initialize particle positions as flattened parameter tensors
            position = torch.cat([param.data.flatten() for param in self.model.parameters()]).to(self.device)
            self.particles.append(position)
            self.pbest_positions.append(position.clone())  # Initialize personal best positions
            self.pbest_losses.append(float('inf'))  # Initialize personal best losses

    def step(self, train_loader):
        """Perform one optimization step."""
        for i in range(self.n_particles):
            # Evaluate loss for each particle
            loss = self._evaluate_loss(self.particles[i], train_loader)

            # Update personal best position and loss
            if loss < self.pbest_losses[i]:
                self.pbest_losses[i] = loss
                self.pbest_positions[i] = self.particles[i].clone()

            # Update global best position and loss
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_position = self.particles[i].clone()

        # Update particle positions
        self._update_positions()

    def _evaluate_loss(self, position, train_loader):
        """Evaluate the loss function for a given particle position."""
        self._set_model_parameters(position)  # Set model parameters to the particle's position
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def _update_positions(self):
        """Update particle positions based on the QPSO algorithm."""
        # Compute the mean best position (mbest)
        mbest = torch.mean(torch.stack(self.pbest_positions), dim=0)

        for i in range(self.n_particles):
            # Generate random numbers
            phi_p_rand = np.random.rand()
            phi_g_rand = np.random.rand()

            # Compute the position for each particle
            p = (self.phi_p * phi_p_rand * self.pbest_positions[i] +
                 self.phi_g * phi_g_rand * self.best_position) / (self.phi_p * phi_p_rand + self.phi_g * phi_g_rand)

            u = np.random.rand()
            new_position = p + np.random.normal(0, 1) * torch.abs(self.particles[i] - mbest) * np.log(1 / u)
            self.particles[i] = new_position.float().to(self.device)

    def _set_model_parameters(self, position):
        """Set model parameters from a flattened position tensor."""
        offset = 0
        for param in self.model.parameters():
            param_size = param.data.numel()
            param.data = position[offset:offset + param_size].reshape(param.data.shape)
            offset += param_size

    def get_best_model(self):
        """Return the model with the best parameters found during optimization."""
        self._set_model_parameters(self.best_position)
        return self.model
