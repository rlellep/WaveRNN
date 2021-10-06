import os
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, SubsetRandomSampler

from utils.dsp import *
from utils import hparams as hp


class RandomImbalancedSampler(Sampler):
    
    def __init__(self, data_source):
        lebel_freq = {}
        for idx in range(len(data_source)):
            label = data_source.items[idx]['language']
            if label in lebel_freq: lebel_freq[label] += 1
            else: lebel_freq[label] = 1

        total = float(sum(lebel_freq.values()))
        weights = [total / lebel_freq[data_source.items[idx]['language']] for idx in range(len(data_source))]
        self._sampler = WeightedRandomSampler(weights, len(weights))

    def __iter__(self):
        return self._sampler.__iter__()

    def __len__(self):
        return len(self._sampler)


class TextToSpeechDataset(Dataset):
    def __init__(self, data_path, metadata):
        self.items = [{'id' : x[0], 'language' : x[2]} for x in metadata]
        self.mel_path = os.path.join(data_path, "mel")
        self.quant_path = os.path.join(data_path, "quant")

    def __getitem__(self, index):
        f_id = self.items[index]['id']
        m = np.load(os.path.join(self.mel_path, f'{f_id}.npy'))
        x = np.load(os.path.join(self.quant_path, f'{f_id}.npy'))
        return m, x

    def __len__(self):
        return len(self.items)


def get_vocoder_datasets(path, batch_size):

    with open(os.path.join(path, 'dataset.pkl'), 'rb') as f:
        metadata = pickle.load(f)
        languages = {}
        train_metadata, test_metadata = [], []
        for i, _, l in metadata:
            add_train = False
            if l in languages:
                languages[l] += 1
                add_train = languages[l] > hp.voc_test_samples // 10
            else:
                add_train = True
                languages[l] = 0
            if add_train: train_metadata += [(i, _, l)]
            else: test_metadata += [(i, _, l)]

    train_dataset = TextToSpeechDataset(path, train_metadata)
    test_dataset = TextToSpeechDataset(path, test_metadata)

    sampler = RandomImbalancedSampler(train_dataset)
    train_set = DataLoader(train_dataset, collate_fn=collate_vocoder, shuffle=False, 
                             sampler=sampler, num_workers=2, batch_size=batch_size, pin_memory=True)

    test_set = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

    return train_set, test_set


def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL':
        y = label_2_float(y.float(), bits)

    return x, y, mels
