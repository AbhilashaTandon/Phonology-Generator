import torch
from autoencoder import AE
from generation import (
    gen_avg_inventory,
    gen_inventory,
    mix_languages,
    gen_rand_inventory,
    negative_lang,
)
from hyperparam import (
    load_data,
    LATENT_FEATURES,
    LAYER_SIZES,
    LR,
    BATCH_SIZE,
    WEIGHT_DECAY,
)
import numpy as np

rng = np.random.default_rng()

saved_models = "saved_models/model.pt"


def get_inverse_percentile(data, value):
    # assumes data is sorted
    first_idx = -1
    last_idx = -1
    for idx, x in enumerate(data):
        if x == value and first_idx < 0:
            first_idx = idx
        if x == value:
            last_idx = idx

    if first_idx < 0:
        return None

    return data[len(data) - 1 - ((first_idx + last_idx) // 2)]


def main():
    languages, phoneme_list, named_languages = load_data()
    phoneme_to_id = {phoneme.ipa: id for id, phoneme in enumerate(phoneme_list)}

    phoneme_counts = sorted([language.num_phonemes() for language in languages])

    # if these two parameters aren't the same as the ones in the saved model, this won't work

    model = AE(
        LATENT_FEATURES,
        LAYER_SIZES,
        LR,
        BATCH_SIZE,
        WEIGHT_DECAY,
        load_from=saved_models,
    )

    # print(named_languages["English"])

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

        print(
            mix_languages(
                model,
                "Pirahã",
                "!Xóõ",
                named_languages,
                phoneme_list,
                phoneme_to_id,
            )
        )

        # for lang in [
        #     "French",
        #     "Spanish",
        #     "English",
        #     "Russian",
        #     "Hawaiian",
        #     "Korean",
        #     "Kabardian",
        #     "Mandarin Chinese",
        #     "Pirahã",
        # ]:
        #     phonemes_in_lang = named_languages[lang].num_phonemes()
        #     inverse_phonemes = get_inverse_percentile(phoneme_counts, phonemes_in_lang)
        #     print(
        #         lang,
        #         negative_lang(
        #             model,
        #             lang,
        #             named_languages,
        #             phoneme_list,
        #             phoneme_to_id,
        #             mean,
        #             inverse_phonemes,
        #         ),
        #     )


if __name__ == "__main__":
    main()
