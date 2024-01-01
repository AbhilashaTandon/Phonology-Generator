import torch
from language import Language, Phoneme


class VAE(torch.nn.Module):
    # features and compressed features
    def __init__(self, num_features, latent_features):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, 480),
            torch.nn.ReLU(),
            torch.nn.Linear(480, 320),
            torch.nn.ReLU(),
            torch.nn.Linear(320, 213),
            torch.nn.ReLU(),
            torch.nn.Linear(213, 142),
            torch.nn.ReLU(),
        )

        self.to_mu = torch.nn.Sequential(
            torch.nn.Linear(142, 95),
            torch.nn.ReLU(),
            torch.nn.Linear(95, latent_features)
        )

        self.to_sigma = torch.nn.Sequential(
            torch.nn.Linear(142, 95),
            torch.nn.ReLU(),
            torch.nn.Linear(95, latent_features)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_features, 95),
            torch.nn.ReLU(),
            torch.nn.Linear(95, 142),
            torch.nn.ReLU(),
            torch.nn.Linear(142, 213),
            torch.nn.ReLU(),
            torch.nn.Linear(213, 320),
            torch.nn.ReLU(),
            torch.nn.Linear(320, 480),
            torch.nn.ReLU(),
            torch.nn.Linear(480, num_features),
            torch.nn.Sigmoid()
        )

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.to_mu(encoded)
        sigma = torch.exp(self.to_sigma(encoded))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 /
                   2).sum() / (sigma.shape[0] * sigma.shape[1])
        decoded = self.decoder(z)

        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, vec):
        decoded = self.decoder(vec)
        return decoded
