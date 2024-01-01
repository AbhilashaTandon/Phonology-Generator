import io
import pstats
import cProfile
import torch
import numpy as np
import pandas as pd
from language import Phoneme, Language
from autoencoder import AE
from generation import gen_avg_inventory, gen_inventory

pwd = "C:\\Users\\abhil\\Documents\\Programming\\Machine Learning\\Phonology Generator\\"

saved_models = pwd + "\\saved_models\\model.pt"


def load_data():
    phoible = pd.read_csv(pwd + "phoible.csv", low_memory=False)
# Data cleaning

    phoible = phoible.drop(
        columns=['SpecificDialect', 'GlyphID', 'Glottocode', 'Source'])  # unneeded cols

    moa_ids = {'stop': 0, 'affricate': 1, 'fricative': 2, 'nasal': 3, 'approximant': 4,
               'vocalic': 5, 'click': 6, 'blend': 7, 'syllabic_cons': 8, 'vowel': 9, 'tone': 10}

    num_moas = len(moa_ids)

    phonemes = {}  # add to dict first, then make list sorted by occurences

    for _, row in phoible.iterrows():
        ipa = row['Phoneme']
        if (ipa not in phonemes):
            new_phoneme = Phoneme(row)
            phonemes[ipa] = new_phoneme
        else:
            phonemes[ipa].occurences += 1

    phoneme_list = sorted(
        phonemes.values(), key=lambda x: x.occurences, reverse=True)

    # get rid of phonemes that are very infrequent
    phoneme_list = [x for x in phoneme_list if x.occurences > 5]

    # for x in phoneme_list:
    #     print("%s\t%.2f\t%s" % (x.ipa, x.poa, x.moa))

    print("%d Phonemes" % len(phoneme_list))

    phoneme_to_id = {phoneme.ipa: id for id,
                     phoneme in enumerate(phoneme_list)}

    languages = {}

    for _, row in phoible.iterrows():
        lang_name = row['LanguageName']
        if (lang_name not in languages):

            new_language = Language(lang_name)
            languages[lang_name] = new_language

        if (row['Phoneme'] in phoneme_to_id):
            current_phoneme = Phoneme(row)

            languages[lang_name].add_phoneme([current_phoneme])

    languages = list(languages.values())  # make it so we index

    return languages, phoneme_list


def train(model, train_data, phoneme_list, phoneme_to_id, loss_function, optimizer, scheduler, epochs, batch_size, verbose=True):
    for epoch in range(epochs):
        avg_loss = 0
        np.random.shuffle(train_data)
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            min_index = i * batch_size
            max_index = min((i+1) * batch_size, len(train_data))
            batch = np.array([train_data[x].get_vector(phoneme_list, phoneme_to_id)
                             for x in range(min_index, max_index)])

            batch = torch.Tensor(
                batch).reshape(-1, len(phoneme_list))

            reconstructed = model(batch)

            mse_loss = loss_function(reconstructed, batch)

            loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updating moving average
            mse_loss_val = mse_loss.detach().item()
            avg_loss += mse_loss_val

        avg_loss /= num_batches
        if (verbose):
            print("Epoch #%d\t%.6f" % (epoch, avg_loss))
        scheduler.step(avg_loss)

    return model, avg_loss


def test(model, test_data, phoneme_list, phoneme_to_id, loss_function):
    with torch.no_grad():
        avg_loss = 0
        for idx, lang in enumerate(test_data):
            input = lang.get_vector(phoneme_list, phoneme_to_id)
            input = torch.Tensor(input.reshape(len(phoneme_list)))

            reconstructed = model(input)

            mse_loss = loss_function(reconstructed, input)

            avg_loss += mse_loss.detach().item()
        return avg_loss / len(test_data)


def post_training(model, languages, phoneme_list, phoneme_to_id, latent_features):
    # get average encoded vector
    with torch.no_grad():
        enc_vector = []
        for i in range(len(languages)):
            inventory_vec = torch.Tensor(
                languages[i].get_vector(phoneme_list, phoneme_to_id))
            encoded = model.encode(inventory_vec).detach().numpy()
            enc_vector.append(encoded)

        enc_vector = np.array(enc_vector)

        mean = np.mean(enc_vector, axis=0)
        stdev = np.std(enc_vector, axis=0)

        for x in range(10):
            inventory = gen_inventory(
                model, mean, stdev, phoneme_list, latent_features)
            print(inventory)

        print("Average Inventory")
        print(gen_avg_inventory(model, mean, phoneme_list, latent_features))


def gen_layers(input_features, latent_features):
    layers = [input_features]

    hidden_features = input_features

    while (hidden_features > 2 * latent_features):
        hidden_features = np.random.randint(latent_features, hidden_features)
        layers.append(hidden_features)

    layers.append(latent_features)

    return layers


def main():
    languages, phoneme_list = load_data()
    phoneme_to_id = {phoneme.ipa: id for id,
                     phoneme in enumerate(phoneme_list)}

    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(
        languages, test_size=.2, random_state=42)
    train_data, dev_data = train_test_split(
        train_data, test_size=.2, random_state=42)

    latent_features = 20  # len of compressed vector

    layer_sizes = [719, 40]

    model = AE(
        latent_features, layer_sizes)  # 20 different parameters
    LR = 1e-3

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR)

    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)

    print("Number of trainable parameters: %d" % num_params)
    # number of trainable parameters

    # checkpoint = torch.load(saved_models)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=.3, mode='min', verbose=True)

    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Validation using Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer

    model, train_loss = train(model, train_data, phoneme_list, phoneme_to_id, loss_function,
                              optimizer, scheduler, epochs=10, batch_size=64, verbose=False)

    test_loss = test(model, dev_data,
                     phoneme_list, phoneme_to_id, loss_function)
    print("%d\t%.4f" % (num_params, test_loss), layer_sizes)

    # torch.save({'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, saved_models)
    # print("MODEL SAVED")

    # post_training(model, languages, phoneme_list,
    #               phoneme_to_id, latent_features)


if __name__ == "__main__":
    main()
