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


class Phoneme():
    def __init__(self, ipa):
        self.ipa = ipa
        self.occurence = 1
        self.moa = ''

    def set_moa(self, row):
        segment_class = row['SegmentClass']
        if (segment_class == 'vowel'):
            self.moa = 'vowel'
        elif (segment_class == 'tone'):
            self.moa = 'tone'
        elif (row['syllabic'] == '+'):
            self.moa = 'syllabic_cons'
        elif (row['consonantal'] == '-'):
            self.moa = 'vocalic'
        elif (row['continuant'] == '+' and row['sonorant'] == '-'):
            self.moa = 'fricative'
        elif (row['continuant'] == '-'):
            self.moa = 'stop'
        elif (row['sonorant'] == '+'):
            self.moa = 'sonorant'
        else:
            self.moa = 'none'


phonemes = {}

for _, row in phoible.iterrows():
    ipa = row['Phoneme']
    if (ipa not in phonemes):
        new_phoneme = Phoneme(ipa)
        new_phoneme.set_moa(row)
    else:
        phonemes[ipa].occurences += 1

phoneme_list = sorted(
    phonemes.values(), key=lambda x: x.occurences, reverse=True)

for x in phoneme_list[:10]:
    print(x.ipa)


class Language():
    def __init__(self, name, lang_id):
        self.id = lang_id
        self.name = name
        self.phoneme_arr = [0 for x in phonemes]
        self.phonemes = []
        # 6 classes, stops frics sons vocalic vowels tones
        self.phoneme_classes = [0, 0, 0, 0, 0, 0]

    def add_phoneme(self, phoneme):  # adds single phoneme to given language
        self.phoneme_arr[phoneme.id] = 1
        self.phonemes.append(phoneme.ipa)
        self.phoneme_classes[phoneme.moa] += 1
