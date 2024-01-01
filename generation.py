import torch
import numpy as np
from language import Language, Phoneme


def gen_inventory(model, mean, stdev, phoneme_list, latent_features):
    with torch.no_grad():
        # assumes encoded inventories follow a normal dist
        variance = np.random.normal(0, 1, size=(
            1, latent_features)) * stdev
        vec = torch.Tensor(np.add(variance, mean))
        decoded = model.decode(vec)
        threshold = np.random.rand() * .25 + .125
        inventory = Language("inventory")
        inventory.from_vector(
            decoded.detach().numpy().reshape(-1), phoneme_list, threshold)
        return inventory


def gen_avg_inventory(model, mean, phoneme_list, latent_features):
    with torch.no_grad():
        vec = torch.Tensor(mean.reshape(1, latent_features))
        decoded = model.decode(vec)
        inventory = Language("Average Inventory")
        inventory.from_vector(
            decoded.detach().numpy().reshape(-1), phoneme_list, .25)
        return inventory
