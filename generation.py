import torch
import numpy as np
from language import Language, Phoneme


def gen_rand_inventory(model, mean, stdev, phoneme_list, latent_features):
    with torch.no_grad():
        # assumes encoded inventories follow a normal dist
        variance = np.random.normal(0, 1, size=(1, latent_features)) * stdev
        vec = torch.Tensor(np.add(variance, mean))
        decoded = model.decode(vec)
        threshold = np.random.rand() * 0.25 + 0.125
        inventory = Language("inventory")
        inventory.from_vector(
            decoded.detach().numpy().reshape(-1), phoneme_list, threshold
        )
        return inventory


def gen_inventory(model, latent_vector, phoneme_list, threshold):
    decoded = model.decode(torch.Tensor(latent_vector))
    inventory = Language("inventory")
    inventory.from_vector(decoded.detach().numpy().reshape(-1), phoneme_list, threshold)
    return inventory


def gen_avg_inventory(model, mean, phoneme_list, latent_features):
    with torch.no_grad():
        vec = torch.Tensor(mean.reshape(1, latent_features))
        decoded = model.decode(vec)
        inventory = Language("Average Inventory")
        inventory.from_vector(decoded.detach().numpy().reshape(-1), phoneme_list, 0.25)
        return inventory


def mix_languages(
    model, lang_name_1, lang_name_2, languages, phoneme_list, phoneme_to_id
):
    if lang_name_1 not in languages or lang_name_2 not in languages:
        return None

    lang_vec_1 = torch.Tensor(
        languages[lang_name_1].get_vector(phoneme_list, phoneme_to_id)
    )
    lang_vec_2 = torch.Tensor(
        languages[lang_name_2].get_vector(phoneme_list, phoneme_to_id)
    )

    size = (
        languages[lang_name_1].num_phonemes() + languages[lang_name_2].num_phonemes()
    ) // 2

    encoded = (model.encode(lang_vec_1) + model.encode(lang_vec_2)) / 2

    decoded = model.decode(encoded)

    inventory = Language("inventory")
    inventory.from_vector_sized(
        decoded.detach().numpy().reshape(-1), phoneme_list, size
    )
    return inventory


def negative_lang(
    model, lang_name, languages, phoneme_list, phoneme_to_id, mean, num_phonemes
):
    if lang_name not in languages:
        return None

    lang_vec = torch.Tensor(
        languages[lang_name].get_vector(phoneme_list, phoneme_to_id)
    )

    encoded = model.encode(lang_vec)

    decoded = model.decode(2 * torch.Tensor(mean) - encoded)

    inventory = Language("inventory")
    inventory.from_vector_sized(
        decoded.detach().numpy().reshape(-1), phoneme_list, num_phonemes
    )
    return inventory
