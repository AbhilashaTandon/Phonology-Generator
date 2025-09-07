# import io
# import pstats
# import cProfile
import torch
import numpy as np
import pandas as pd
from language import Phoneme, Language
from autoencoder import AE
from sklearn.model_selection import GridSearchCV

# LATENT_FEATURES = 10  # len of compressed vector
# LAYER_SIZES = [719, 40]
# BATCH_SIZE = 64
# LR = 1e-3


saved_models = "saved_models/model.pt"


def load_data():
    phoible = pd.read_csv("phoible.csv", low_memory=False)
    # Data cleaning

    phoible = phoible.drop(
        columns=["SpecificDialect", "GlyphID", "Glottocode", "Source"]
    )  # unneeded cols

    # moa_ids = {'stop': 0, 'affricate': 1, 'fricative': 2, 'nasal': 3, 'approximant': 4,
    #            'vocalic': 5, 'click': 6, 'blend': 7, 'syllabic_cons': 8, 'vowel': 9, 'tone': 10}

    # num_moas = len(moa_ids)

    phonemes = {}  # add to dict first, then make list sorted by occurences

    for _, row in phoible.iterrows():
        ipa = row["Phoneme"]
        if ipa not in phonemes:
            new_phoneme = Phoneme(row)
            phonemes[ipa] = new_phoneme
        else:
            phonemes[ipa].occurences += 1

    phoneme_list = sorted(phonemes.values(), key=lambda x: x.occurences, reverse=True)

    # get rid of phonemes that are very infrequent
    phoneme_list = [x for x in phoneme_list if x.occurences > 5]

    # for x in phoneme_list:
    #     print("%s\t%.2f\t%s" % (x.ipa, x.poa, x.moa))

    print("%d Phonemes" % len(phoneme_list))

    phoneme_to_id = {phoneme.ipa: id for id, phoneme in enumerate(phoneme_list)}

    languages = {}

    for _, row in phoible.iterrows():
        lang_name = row["LanguageName"]
        if lang_name not in languages:
            new_language = Language(lang_name)
            languages[lang_name] = new_language

        if row["Phoneme"] in phoneme_to_id:
            if row["Marginal"] == "True":
                continue
            current_phoneme = Phoneme(row)

            languages[lang_name].add_phoneme([current_phoneme])

    language_values = list(languages.values())  # make it so we index

    return language_values, phoneme_list, languages


LATENT_FEATURES = 10  # len of compressed vector
LAYER_SIZES = [719, 200]
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5


def main():
    languages, phoneme_list, _ = load_data()
    phoneme_to_id = {phoneme.ipa: id for id, phoneme in enumerate(phoneme_list)}

    from sklearn.model_selection import train_test_split

    train_data, _ = train_test_split(languages, test_size=0.2, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=42)

    model = AE(
        LATENT_FEATURES,
        LAYER_SIZES,
        LR,
        BATCH_SIZE,
        WEIGHT_DECAY,
    )  # 20 different parameters

    loss_function = torch.nn.MSELoss()

    model.train(
        train_data,
        dev_data,
        phoneme_list,
        phoneme_to_id,
        loss_function,
        epochs=100,
        verbose=True,
        save_to=saved_models,
    )

    model.post_training(languages, phoneme_list, phoneme_to_id, LATENT_FEATURES)


if __name__ == "__main__":
    main()


# consider trying lasso regression (L1 normalization)
