import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
from icecream import ic

from DL.pytorch_version.models.DSU import DistributionUncertainty


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.acti = nn.LeakyReLU

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = self.acti()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = self.acti()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = self.acti()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, params, configuration_run):
        super(TemporalConvNet, self).__init__()

        try:
            self.len_size = params['padding_size']
        except KeyError:
            self.len_size = params['signal_size']

        self.window_len = params['window_size']
        self.num_window = int(self.len_size / self.window_len)

        num_channels = params['n_blocks'] * [params['n_filter_initial']]
        num_inputs = 1
        kernel_size = params['kernel_size']
        dropout = params['dropout']

        e1_module, e2_module = [], []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            if i < num_levels / 2:
                e1_module += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                            padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            else:
                e2_module += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                            padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        if configuration_run == 'DSU':
            e1_module += [DistributionUncertainty()]
            e2_module += [DistributionUncertainty()]


        self.e1_model = nn.Sequential(*e1_module)
        self.e2_model = nn.Sequential(*e2_module)

        classifier_layers = [nn.AdaptiveAvgPool1d(output_size=1),
                             nn.Flatten(),
                             nn.Linear(in_features=num_channels[-1] * self.num_window, out_features=256),
                             nn.BatchNorm1d(num_features=256), nn.LeakyReLU(), nn.Dropout(p=0.3),
                             nn.Linear(in_features=256, out_features=1)
                             ]
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        batch_size = x.shape[0]

        x = torch.unsqueeze(x, 1)
        x = torch.reshape(x, shape=(batch_size*self.num_window, 1, self.window_len))

        e1 = self.e1_model(x)
        e2 = self.e2_model(e1)

        e1 = torch.reshape(e1, shape=(batch_size, -1, self.window_len))
        e2 = torch.reshape(e2, shape=(batch_size, -1, self.window_len))

        out = self.classifier(e2)
        out = torch.squeeze(out)

        return e1, e2, out
