import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------

class MDN_RNN(nn.Module):

    def __init__(self, z_dim, action_dim, hidden_dim, num_gaussians):

        super().__init__()

        self.input_dim = z_dim + action_dim
        self.rnn = nn.LSTM(self.input_dim, hidden_dim, bias=True, batch_first=True)

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        
        self.output_dim = num_gaussians * (2*z_dim + 1)

        self.mdn = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, z, a, hidden=None):

        x = torch.concat([z, a], dim=-1) # (batch, seq_len, z_dim + action_dim)

        h_seq, hidden = self.rnn(x, hidden) # (batch, seq_len, hidden_dim)

        mdn_out = self.mdn(h_seq) # (batch, seq_len, K*(2*z_dim + 1))

        K = self.num_gaussians
        D = self.z_dim

        # split mixture parameters
        pi_logits = mdn_out[..., :K]
        mu_raw = mdn_out[..., K:K + K*D]
        sigma_raw = mdn_out[..., K + K*D:]

        # mixture weights
        pi = torch.softmax(pi_logits, dim=-1)

        # reshape means
        mu = mu_raw.view(mu_raw.size(0), mu_raw.size(1), K, D)

        # reshape stds
        sigma = sigma_raw.view(sigma_raw.size(0), sigma_raw.size(1), K, D)

        # enforce positive std
        sigma = torch.exp(sigma)

        return pi, mu, sigma, hidden




