import torch
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

phoible = pd.read_csv(
    "C:\\Users\\abhil\\Documents\\Programming\\Machine Learning\\Phonology Generator\\phoible.csv", low_memory=False)
# Data cleaning

phoible = phoible.drop(
    columns=['SpecificDialect', 'GlyphID', 'Glottocode', 'Source'])  # unneeded cols

moa_ids = {'stop': 0, 'fricative': 1, 'nasal': 2, 'approximant': 3,
           'vocalic': 4, 'click': 5, 'blend': 6, 'syllabic_cons': 7, 'vowel': 8, 'tone': 9}

num_moas = len(moa_ids)


class Phoneme():
    def __init__(self, row):
        self.ipa = row['Phoneme']  # ipa symbol, is a string
        self.occurences = 1  # we start with 1
        self.set_moa(row)

    def set_moa(self, row):  # uses features given in PHOIBLE to assign a class
        segment_moa = row['SegmentClass']
        if (segment_moa == 'vowel'):
            self.moa = 'vowel'
        elif (segment_moa == 'tone'):
            self.moa = 'tone'
        elif (row['click'] == '+'):
            self.moa = 'click'
        elif (row['syllabic'] == '+'):  # if syllabic but not vowel
            self.moa = 'syllabic_cons'
        elif (row['consonantal'] == '-'):  # if not consonantal but not vowel
            self.moa = 'vocalic'
        elif (row['continuant'] == '+' and row['sonorant'] == '-'):
            self.moa = 'fricative'
        elif (row['sonorant'] == '+' and row['nasal'] == '+'):
            self.moa = 'nasal'
        elif (row['continuant'] == '-'):  # not continuant but consonant
            self.moa = 'stop'
        elif (row['approximant'] == '+'):
            self.moa = 'approximant'
        else:
            self.moa = 'blend'


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

# get rid of phonemes that occur in 10 or less langs
phoneme_list = [x for x in phoneme_list if x.occurences > 10]

print(len(phoneme_list))

phoneme_to_id = {phoneme.ipa: id for id, phoneme in enumerate(phoneme_list)}


class Language():
    def __init__(self, name, lang_id):
        self.id = lang_id
        self.name = name
        self.phoneme_arr = [0 for x in phoneme_list]
        self.phonemes = []
        # 6 classes, stops frics sons vocalic vowels tones
        self.phoneme_moas = [0 for x in moa_ids]

    def add_phoneme(self, phoneme_id):  # adds single phoneme to given language
        phoneme = phoneme_list[phoneme_id]
        self.phoneme_arr[phoneme_id] = 1
        self.phonemes.append(phoneme.ipa)
        self.phoneme_moas[moa_ids[phoneme.moa]] += 1

    def get_vector(self):
        return self.phoneme_arr


languages = {}

for _, row in phoible.iterrows():
    lang_id = row['InventoryID']
    if (lang_id not in languages):

        new_language = Language(row['LanguageName'], lang_id)
        languages[lang_id] = new_language

    if (row['Phoneme'] in phoneme_to_id):
        current_phoneme = phoneme_to_id[row['Phoneme']]

        languages[lang_id].add_phoneme(current_phoneme)

languages = list(languages.values())  # make it so we index


class AE(torch.nn.Module):
    # features and compressed features
    def __init__(self, num_features, num_comp_phon_features):
        super().__init__()

        self.comp_vec = np.zeros(
            (1, num_comp_phon_features))  # compressed vector

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, num_comp_phon_features)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_comp_phon_features, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, num_features),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        self.comp_vec = encoded
        decoded = self.decoder(encoded)

        return decoded

    def decode(self, vec):
        decoded = self.decoder(vec)
        return decoded


num_comp_phon_features = 20

model = AE(len(phoneme_list),
           num_comp_phon_features)  # 10 different classes

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=7e-4,
                             weight_decay=1e-8)


def vec_to_inventory(phon_vec, threshold):
    inventory = {}
    for moa_name in moa_ids:
        inventory[moa_name] = []
    for idx, val in enumerate(phon_vec):
        if (val > threshold):
            phoneme = phoneme_list[idx]
            inventory[phoneme.moa].append(phoneme.ipa)
    return inventory


def gen_inventory(model):
    vec = torch.Tensor(np.random.normal(
        0, 2, size=(1, num_comp_phon_features)))
    decoded = model.decode(vec)
    threshold = np.random.rand() * .25 + .25
    return vec_to_inventory(decoded.detach().numpy().reshape(-1), threshold)


epochs = 10
batch_size = 4
outputs = []
losses = []

for epoch in range(epochs):
    losses = []
    avg_loss = 0
    np.random.shuffle(languages)
    for i in range(len(languages) // batch_size):
        min_index = i * batch_size
        max_index = min((i+1) * batch_size, len(languages))
        batch = np.array([languages[x].get_vector()
                         for x in range(min_index, max_index)])

        batch = torch.Tensor(
            batch).reshape(-1, len(phoneme_list))

        # Output of Autoencoder
        reconstructed = model(batch)

        # Calculating the loss function
        loss = loss_function(reconstructed, batch)

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        loss_val = loss.detach().item()
        losses.append(loss_val)
        if (avg_loss == 0):
            avg_loss = loss_val
        else:
            avg_loss = .99 * avg_loss + .01 * loss_val

        if (i % 100 == 0):
            print(i * batch_size, avg_loss)

plt.plot([i for i, _ in enumerate(losses)], losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

for x in range(10):
    inventory = gen_inventory(model)
    for x in inventory.values():
        if (len(x) > 0):
            print(' '.join(x))
    print()
    print()
