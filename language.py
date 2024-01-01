import numpy as np


class Phoneme():
    def __init__(self, row):
        self.ipa = row['Phoneme']  # ipa symbol, is a string
        self.occurences = 1  # we start with 1
        self.set_moa(row)
        self.set_poa(row)

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
        elif (row['delayedRelease'] == '+'):
            self.moa = 'affricate'
        else:
            self.moa = 'blend'

    def set_poa(self, row):  # gives a number where consonants at the front of the mouth come first
        self.poa = 0

        vowel_pos = 0
        if (row['labial'] == '+'):
            self.poa = (1.5 if (row['round'] == '+') else 1)
        elif (row['labiodental'] == '+'):
            self.poa = 2
        elif (row['coronal'] == '+'):
            self.poa = 3
            if (row['anterior'] == '-'):
                self.poa += .5
            if (row['distributed'] == '-'):
                self.poa += .2
        elif (row['dorsal'] == '+'):
            self.poa = 5
            if (row['front'] == '+'):
                vowel_pos -= .2
            if (row['back'] == '+'):
                vowel_pos += .2
            if (row['high'] == '+'):
                vowel_pos -= .4
            if (row['low'] == '+'):
                vowel_pos += .4
            if (row['tense'] == '+'):
                vowel_pos += .1
        if (row['retractedTongueRoot'] == '+'):
            vowel_pos += .1
        if (row['advancedTongueRoot'] == '+'):
            vowel_pos -= .1
        if (row['SegmentClass'] == 'vowel' or self.moa == 'vocalic'):  # idk why this works but it does
            self.poa -= vowel_pos
            self.poa = -self.poa
        else:
            self.poa += vowel_pos
        if (row['periodicGlottalSource'] == '+'):
            self.poa += 1e-3
        self.poa += len(self.ipa) * .01

    def __str__(self):
        return self.ipa


class Language():
    def __init__(self, name):
        self.name = name
        self.phonemes = []
        # 6 classes, stops frics sons vocalic vowels tones

    def add_phoneme(self, phonemes):  # adds single phoneme to given language
        for x in phonemes:
            self.phonemes.append(x)

    def get_vector(self, phoneme_list, phoneme_to_id):
        out_vector = np.zeros(len(phoneme_list))  # blank vector
        for phoneme in self.phonemes:
            out_vector[phoneme_to_id[phoneme.ipa]] = 1.  # one hot encoding
        return np.array(out_vector)

    def get_inventory(self):
        inventory = {}
        for phoneme in self.phonemes:  # add to separate list for each moa
            if (phoneme.moa in inventory):
                inventory[phoneme.moa].append(phoneme)
            else:
                inventory[phoneme.moa] = [phoneme]
        for moa in inventory:
            # sort phonemes by poa, labial to glottal
            inventory[moa] = sorted(inventory[moa], key=lambda x: x.poa)
        return inventory

    def from_vector(self, vec, phoneme_list, threshold):
        self.phonemes = []  # clears inventory
        for val, phoneme in zip(vec, phoneme_list):
            if (val > threshold):
                self.add_phoneme([phoneme])

    def __str__(self):
        inv = self.get_inventory()
        out_str = ""
        for moa in inv:
            for phoneme in inv[moa]:
                out_str += str(phoneme) + " "
            out_str += "\n"
        return out_str
