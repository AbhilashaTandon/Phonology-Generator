import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import torch
import numpy as np
import numpy as np
import pandas as pd

phoible = pd.read_csv(
    "C:\\Users\\abhil\\Documents\\Programming\\Machine Learning\\Phonology Generator\\phoible.csv")
# Data cleaning

phoible = phoible.drop(
    columns=['SpecificDialect', 'GlyphID', 'Glottocode', 'Source'])

phonemes = sorted(list(set(phoible['Phoneme'])))
languages = sorted(list(set(phoible['LanguageName'])))

print(len(languages))

occurences = [0 for x in phonemes]

features = {}
moas = {}

phoneme_to_id = {phoneme: id for id, phoneme, in enumerate(phonemes)}
language_to_id = {language: id for id, language in enumerate(languages)}

for idx, row in phoible.iterrows():
    feature = row['SegmentClass']
    phoneme_class = ''
    if (feature == 'consonant'):
        moa = (row['consonantal'] == '+',
                            row['sonorant'] == '+',
                            row['continuant'] == '+')
        if (not moa[0]):
            phoneme_class = 'vocalic'
        elif (not moa[1] and moa[2]):
            return 4
        elif (not moa[1] and not moa[2]):
            return 5
        elif (moa[1]):
            return 3
    elif (feature == 'vowel'):
        return 1
    elif (feature == 'tone'):
        return 0
    occurences[phoneme_to_id[row['Phoneme']]] += 1

phonemes = sorted(
    phonemes, key=lambda x: occurences[phoneme_to_id[x]], reverse=True)

occurences = sorted(occurences, reverse=True)


for idx, num in enumerate(occurences):
    if (num < 2):
        occurences[idx] = 0
        phonemes[idx] = '_'

occurences = [occurence for occurence in occurences if occurence > 0]
phonemes = [phoneme for phoneme in phonemes if phoneme != '_']

phoneme_to_id = {phoneme: id for id, phoneme, in enumerate(phonemes)}


instance_list = list(zip(phoible['Phoneme'], phoible['LanguageName']))
for idx, (phoneme, _) in enumerate(instance_list):
    if (phoneme not in phonemes):
        instance_list[idx] = 0

instance_list = [i for i in instance_list if i != 0]

phoneme_idxs = np.array([phoneme_to_id[x[0]] for x in instance_list])
language_idxs = np.array([language_to_id[x[1]] for x in instance_list])

data = [1 for i in range(len(phoneme_idxs))]

inventories = coo_matrix(
    (data, (language_idxs, phoneme_idxs)), dtype=np.float64)

arr_inventories = inventories.toarray()

num_in_class = 

def classify(phoneme):
    


class AE(torch.nn.Module):
    def __init__(self, num_features, num_comp_features):  # features and compressed features
        super().__init__()

        self.comp_vec = np.zeros(
            (1, num_comp_features))  # compressed vector

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, num_comp_features)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_comp_features, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, num_features),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        self.comp_vec = encoded
        decoded = self.decoder(encoded)
        return decoded


num_comp_features = 10

model = AE(len(phonemes), num_comp_features)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-4,
                             weight_decay=1e-8)


def vec_to_inventory(vec, cons_size, vowel_size, tone_size):
    consonants = []
    stops = []
    fricatives = []
    sonorants = []
    vocalic = []
    vowels = []
    tones = []

    sorted_phonemes = sorted(
        phonemes, key=lambda x: vec[phoneme_to_id[x]], reverse=True)

    for phoneme in sorted_phonemes[:300]:  # pick 300 most likely phonemes
        feature = features[phoneme]

        if (feature == 'consonant'):
            consonants.append(phoneme)
        elif (feature == 'vowel'):
            vowels.append(phoneme)
        elif (feature == 'tone'):
            tones.append(phoneme)
        else:
            print(phoneme, feature)

    consonants = consonants[:cons_size]  # pick phonemes according to size
    vowels = vowels[:vowel_size]
    tones = tones[:tone_size]

    for phoneme in consonants:
        moa = moas[phoneme]
        if (not moa[0]):
            vocalic.append(phoneme)
        elif (not moa[1] and moa[2]):
            fricatives.append(phoneme)
        elif (not moa[1] and not moa[2]):
            stops.append(phoneme)
        elif (moa[1]):
            sonorants.append(phoneme)

    return stops, fricatives, sonorants, vocalic, vowels, tones


epochs = 5
batch_size = 16
outputs = []

for epoch in range(epochs):
    losses = []
    i = 0
    while (i * batch_size < len(arr_inventories)):
        batch = arr_inventories[i *
                                batch_size:min(len(arr_inventories), (i+1)*batch_size)]

        inv = torch.Tensor(batch.reshape(-1, len(phonemes)))

        # Output of Autoencoder
        reconstructed = model(inv)

        # Calculating the loss function
        loss = loss_function(reconstructed, inv)

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss.detach())

        if (i % 10 == 1):
            losses = np.array(losses)
            print(i * batch_size, (np.sum(losses)/len(losses)))
            losses = []

        i += 1


def gen_inventory(model, num_comp_features):
    random_inv = model.decoder(torch.Tensor(np.random.normal(
        0, 100., size=(1, num_comp_features))))

    cons_size = np.random.normal(loc=22, scale=10)

    if (cons_size < 6):
        cons_size += 6
    if (cons_size > 30):
        cons_size = cons_size ** 1.25

    cons_size = int(cons_size)

    cv_ratio = np.random.normal(5. - np.log(cons_size), scale=1)

    if (cv_ratio < .9):
        cv_ratio += 1

    vow_size = int(cons_size / cv_ratio)

    ton_size = np.random.geometric(p=.58)

    if (ton_size == 1):
        ton_size = 0  # 1 tone and 0 tones are equivalent so might as well not print them

    print(cons_size, vow_size, ton_size)

    return vec_to_inventory(random_inv.reshape(-1), cons_size, vow_size, ton_size)


for x in range(10):  # generate 10 random inventories
    stops, fricatives, sonorants, vocalic, vowels, tones = gen_inventory(
        model, num_comp_features)
    print(stops)
    print(fricatives)
    print(sonorants)
    print(vocalic)
    print(vowels)
    print(tones)
    print()
    print()
