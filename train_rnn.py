import torch
import math
from model import MDN_RNN


def mdn_loss(pi, mu, sigma, z_next):

    z_next = z_next.unsqueeze(2)

    log_prob = -0.5 * (
        ((z_next - mu) / sigma) ** 2 +
        2 * torch.log(sigma) +
        math.log(2 * math.pi)
    )

    log_prob = log_prob.sum(dim=-1)

    log_prob = log_prob + torch.log(pi + 1e-8)

    log_prob = torch.logsumexp(log_prob, dim=-1)

    loss = -log_prob.mean()

    return loss


def train(model, dataloader, optimizer, device):

    model.train()

    total_loss = 0

    for z, a in dataloader:

        z = z.to(device)
        a = a.to(device)

        z_input = z[:, :-1]
        z_next = z[:, 1:]
        a_input = a[:, :-1]

        optimizer.zero_grad()

        pi, mu, sigma, _ = model(z_input, a_input)

        loss = mdn_loss(pi, mu, sigma, z_next)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_dim = 32
    action_dim = 3
    hidden_dim = 256
    num_gaussians = 5

    model = MDN_RNN(z_dim, action_dim, hidden_dim, num_gaussians).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50

    for epoch in range(num_epochs):

        loss = train(model, dataloader, optimizer, device)

        print(f"Epoch {epoch} | Loss: {loss:.4f}")