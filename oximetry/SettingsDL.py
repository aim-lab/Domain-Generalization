import random
import pandas as pd
import os

from utils.help_classes import WrongParameter


class GetDict:
    def __init__(self, model_name):
        self.model_name = model_name

    @staticmethod
    def sample_augmentations():
        aug = ['permutation_augmentation', 'scaling_augmentation', 'rotation_augmentation',
               'magnitude_warp_augmentation', 'time_warp_augmentation', 'data_augmentation', 'jitter_augmentation']

        dict_aug = {}
        for curr_aug in aug:
            # dict_aug[curr_aug] = random.choice([True, False])
            dict_aug[curr_aug] = False

        dict_aug['jitter_augmentation'] = random.choice([True, False])
        return dict_aug

    def get_dict_params(self):
        if self.model_name == 'duplo':
            return {
                'drop': [0.1, 0.2, 0.3, 0.4, 0.5],
                'n_filters_raw': [4, 8, 16, 32],
                'n_filters_stft': [4, 8, 16, 32],
                'n_features_raw': [16, 32, 64, 128],
                'n_features_stft': [16, 32, 64, 128],
                'hidden_dim': [16, 32, 64, 128],
                'kernel_size_raw': [3, 5, 7, 9, 11, 13, 25],
                'kernel_size_stft': [3, 5, 7, 9, 11, 13, 25],
            }
        elif self.model_name == 'resnet':
            return {
                'base_filters': [2, 4, 8, 16],
                'kernel_size': [3, 5, 7, 9, 11],
                'stride': [1, 3, 5],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'n_block': [4, 5, 6, 7, 8, 9, 12]
            }
        elif self.model_name == 'TCN':
            return {
                'n_blocks': [4, 5, 6, 7, 8, 9, 10, 14],
                'n_filter_initial': [8, 16, 32],
                'kernel_size': [3, 5, 7, 9, 11, 13],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        else:
            raise WrongParameter('self.model_name must be in {duplo, resnet, TCN}')

    @staticmethod
    def sample_data_params():
        padding_size_it = [21600, 25200, 28800]
        signal_size_it = [18000, 21600]
        window_size_it = [1200, 1800, 3600]

        dict_data_params = {
            # 'padding_size': random.choice(padding_size_it),
            'signal_size': random.choice(signal_size_it),
            'window_size': random.choice(window_size_it),
        }
        return dict_data_params

    @staticmethod
    def retrieve_dict_run(filename):
        df_file = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/DL/pytorch_version/data_saved/',
                                           filename))
        dict_params = df_file.iloc[0].to_dict()

        dict_params['batch_size'] = int(dict_params['batch_size'])
        return dict_params
