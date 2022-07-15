import os
import torch
from torch import nn
import torch.nn.functional as F
from icecream import ic


class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.ni = ni
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """

    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))

        layers.append(nn.MaxPool1d(kernel_size=2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class DuploClassifier(nn.Module):
    def __init__(self, raw_ni, fft_ni, no, params):
        super().__init__()
        pad = 'same'
        stride = 1

        drop = params['drop']
        n_filters = params['n_filters_raw']

        print(params)
        self.raw = nn.Sequential(
            SepConv1d(raw_ni, n_filters, params['kernel_size_raw'], stride, pad, drop=drop),
            SepConv1d(n_filters, n_filters*2, params['kernel_size_raw'], stride, pad, drop=drop),
            SepConv1d(n_filters*2, n_filters*4, params['kernel_size_raw'], stride, pad, drop=drop),
            SepConv1d(n_filters*4, n_filters*8, params['kernel_size_raw'], stride, pad),
            Flatten(),
            nn.Dropout(drop), nn.Linear(1575*n_filters*8, 256), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(256, params['n_features_raw']), nn.ReLU(inplace=True))

        n_filters = params['n_filters_stft']
        self.fft = nn.Sequential(
            SepConv1d(fft_ni, n_filters, params['kernel_size_stft'], stride, pad, drop=drop),
            SepConv1d(n_filters, n_filters*2, params['kernel_size_stft'], stride, pad, drop=drop),
            SepConv1d(n_filters*2, n_filters*4, params['kernel_size_stft'], stride, pad, drop=drop),
            SepConv1d(n_filters*4, n_filters*4, params['kernel_size_stft'], stride, pad, drop=drop),
            SepConv1d(n_filters*4, n_filters*8, params['kernel_size_stft'], stride, pad),
            Flatten(),
            nn.Dropout(drop), nn.Linear(24*n_filters*8, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(64, params['n_features_stft']), nn.ReLU(inplace=True))

        self.out = nn.Sequential(
            nn.Linear(params['n_features_stft'] + params['n_features_raw'], params['hidden_dim']),
            nn.ReLU(inplace=True), nn.Linear(params['hidden_dim'], no))

    def forward(self, t_raw, t_fft):
        t_raw = torch.unsqueeze(t_raw, 1)

        raw_out = self.raw(t_raw)
        fft_out = self.fft(t_fft)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(t_in)

        out = torch.squeeze(out)
        return out
