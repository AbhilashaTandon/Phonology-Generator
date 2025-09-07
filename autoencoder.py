import torch
from language import Language, Phoneme


class AE(torch.nn.Module):
    # features and compressed features
    def __init__(self, latent_features, layers):
        super().__init__()

        self.encoder = torch.nn.Sequential()

        self.decoder = torch.nn.Sequential()

        for x, y in zip(layers[:-1], layers[1:]):
            self.encoder.append(torch.nn.Linear(x, y))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(layers[-1], latent_features))

        self.decoder.append(torch.nn.Linear(latent_features, layers[-1]))

        layers = layers[::-1]
        for x, y in zip(layers[:-1], layers[1:]):
            self.decoder.append(torch.nn.ReLU())
            self.decoder.append(torch.nn.Linear(x, y))
        self.decoder.append(torch.nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, vec):
        decoded = self.decoder(vec)
        return decoded
