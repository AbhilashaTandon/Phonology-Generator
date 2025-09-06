import torch
from autoencoder import AE
from train import load_data, post_training

saved_models = "saved_models/model.pt"


def main():
    languages, phoneme_list = load_data()
    phoneme_to_id = {phoneme.ipa: id for id, phoneme in enumerate(phoneme_list)}

    latent_features = 20  # len of compressed vector

    layer_sizes = [719, 200, 40]

    # if these two parameters aren't the same as the ones in the saved model, this won't work

    model = AE(latent_features, layer_sizes)

    checkpoint = torch.load(saved_models)
    model.load_state_dict(checkpoint["model_state_dict"])

    post_training(model, languages, phoneme_list, phoneme_to_id, latent_features)


if __name__ == "__main__":
    main()
