import torch
from autoencoder import AE
from generation import gen_avg_inventory, gen_inventory, mix_languages
from train import load_data, LATENT_FEATURES, LAYER_SIZES
import numpy as np

rng = np.random.default_rng()

saved_models = "saved_models/model.pt"


def main():
    languages, phoneme_list, named_languages = load_data()
    phoneme_to_id = {phoneme.ipa: id for id, phoneme in enumerate(phoneme_list)}

    # if these two parameters aren't the same as the ones in the saved model, this won't work

    model = AE(LATENT_FEATURES, LAYER_SIZES)

    # print(named_languages["English"])

    checkpoint = torch.load(saved_models)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        enc_vector = []
        for i in range(len(languages)):
            inventory_vec = torch.Tensor(
                languages[i].get_vector(phoneme_list, phoneme_to_id)
            )
            encoded = model.encode(inventory_vec).detach().numpy()
            enc_vector.append(encoded)

        enc_vector = np.array(enc_vector)

        mean = np.mean(enc_vector, axis=0)
        stdev = np.std(enc_vector, axis=0)

        stdevs = [-99, 99]

        # for x in range(latent_features):
        #     basis_vector = mean.copy()
        #     basis_vector[x] += stdev[x] * 5
        #     threshold = rng.poisson(13) / 100
        #     print(threshold)

        #     inventory = gen_inventory(model, basis_vector, phoneme_list, threshold)
        #     print(inventory)

        for idx in range(LATENT_FEATURES):
            for stdev_val in stdevs:
                vector = mean.copy() + stdev[idx] * stdev_val
                print(idx, stdev_val)
                print(gen_inventory(model, vector, phoneme_list, 0.125))

        # print("Average Inventory")
        # print(gen_avg_inventory(model, mean, phoneme_list, latent_features))

        # print(
        #     mix_languages(
        #         model,
        #         "French",
        #         "Spanish",
        #         named_languages,
        #         phoneme_list,
        #         phoneme_to_id,
        #     )
        # )


if __name__ == "__main__":
    main()
