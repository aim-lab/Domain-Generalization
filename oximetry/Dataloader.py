from torch.utils.data import Dataset
import torch
import numpy as np
import librosa
from icecream import ic


class OximetryDataset(Dataset):
    def __init__(self, x, y, model_name, params, mean, std, dataset_label):
        super().__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.dataset_label = dataset_label.copy()

        self.normalize(mean, std)

        self.model_name = model_name
        self.params = params

    def __len__(self):
        return self.x.shape[0]

    def normalize(self, mean, std):
        self.x = (self.x - mean) / std

    @staticmethod
    def convert_to_tensor(x, y, dataset_label):
        x = torch.from_numpy(x).requires_grad_(True).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        dataset_label = torch.from_numpy(dataset_label).type(torch.FloatTensor)

        return x, y, dataset_label

    def __getitem__(self, idx):
        x = self.x[idx]
        y = np.array(self.y[idx], dtype=np.float)
        dataset_label = np.array(self.dataset_label[idx], dtype=np.float)

        if self.model_name == 'duplo':
            stft = librosa.stft(x, window=self.params['window_stft'], n_fft=self.params['n_fft'],
                                win_length=self.params['win_length'])
            stft = 10. * np.log10(np.abs(stft) + 1e-10)
            stft = torch.from_numpy(stft).requires_grad_(True).type(torch.FloatTensor)

            x, y, dataset_label = self.convert_to_tensor(x, y, dataset_label)
            return x, stft, y, dataset_label

        x, y, dataset_label = self.convert_to_tensor(x, y, dataset_label)
        return x, y, dataset_label
