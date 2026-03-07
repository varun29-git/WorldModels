import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, z_dim, hidden_dim, action_dim):
        super().__init__()
        
        # Controller maps [z, h] -> action
        self.fc = nn.Linear(z_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        """
        z: latent vector from VAE (batch_size, z_dim)
        h: hidden state from MDN-RNN (batch_size, hidden_dim)
        """
        x = torch.cat([z, h], dim=1)
        out = self.fc(x)

        # Assuming action space for CarRacing: [steering, gas, brake]
        # steering is in [-1, 1], gas in [0, 1], brake in [0, 1]
        steering = torch.tanh(out[:, 0:1])
        gas = torch.sigmoid(out[:, 1:2])
        brake = torch.sigmoid(out[:, 2:3])

        return torch.cat([steering, gas, brake], dim=1)
