import sys

sys.path.append('/home/jeremy.levy/Jeremy/copd_osa')

import torch
from tqdm import tqdm
import time
from torch import nn
from icecream import ic
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import utils.graphics as graph
from torch.utils.tensorboard import SummaryWriter

from DL.Main import Main
from DL.SettingsDL import ParamSearch_Duplo
from DL.pytorch_version.Dataloader import OximetryDataset
from DL.pytorch_version.ResNet import ResNet1D
from DL.util_funcs import compute_metrics_osa
from utils.utils_func import save_dict
from DL.pytorch_version.Duplo import DuploClassifier
from DL.pytorch_version.TCN import TemporalConvNet
from utils.help_classes import WrongParameter
from DL.Dataloader.DataGeneratorFullSignal import merge_generator
from DL.pytorch_version.Custom_Loss import CustomLoss


def sample_from_dict(dict_params_it):
    dict_params = {}
    for key in dict_params_it:
        dict_params[key] = random.choice(dict_params_it[key])
    return dict_params


class Main_Pytorch(Main):
    def __init__(self, short_sample, device_ids, num_epochs, model_name):
        super().__init__(type_run='', model_name=model_name, epoch_1plot=0, epoch_1=0, epoch_2=0,
                         short_sample=short_sample)

        self.device_ids = device_ids
        self.num_epochs = num_epochs
        self.epochs_without_improvement = 0

        self.saved_path = 'data_saved'
        os.makedirs(self.saved_path, exist_ok=True)
        self.ticks_fontsize, self.fontsize, self.letter_fontsize = 15, 15, 15

        self.params_it = {'model_name': model_name,

                          # Learning process
                          'regularization_weight': 0.0005, 'learning_rate': 0.0005, 'batch_size': 64,

                          # data
                          'sampling': False, 'apply_median_spo2': False, 'all_metadata': 'meta_pobm',
                          'features_ss': False, 'normalization': False, 'rocket_features': False,
                          'double_channel': False, 'loss_osa': 'MSE', 'loss_copd': 'FocalLoss_digit_false',
                          'osa_regression': True, 'full_signal': True, 'signal_size': 18000, 'padding_size': 18000,
                          'transform_ahi': None, 'activation_function_copd': None, 'activation_function_osa': None,

                          # STFT
                          'window_stft': 'hamming', 'n_fft': 1024, 'win_length': 128,

                          # Split
                          'train_size_UHV_pc': 100, 'train_size_SHHS_pc': 0.9, 'train_size_WSC_pc': 100,
                          'train_size_CFS_pc': 100, 'train_size_SHHS2_pc': 0.7, 'train_size_numom_pc': 100,
                          }

        # device = torch.device('cuda:' + str(self.device_ids) if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print(f"Using {self.device} device")

        self.loss = CustomLoss(device=self.device, regularization_weight=self.params_it['regularization_weight'])

    def plot_output_model(self, y_pred, y_test, epoch):
        y_pred = torch.cat(y_pred, 0).detach().numpy().flatten()
        y_test = torch.cat(y_test, 0).numpy().flatten()

        fig, axes = graph.create_figure(subplots=(1, 1), figsize=(8, 8), sharey=False)

        axes[0][0].hist(y_pred, label='y_pred', alpha=0.5)
        axes[0][0].hist(y_test, label="y_test", alpha=0.5)

        graph.complete_figure(fig, axes, put_legend=[[True]],
                              xticks_fontsize=self.ticks_fontsize, yticks_fontsize=self.ticks_fontsize,
                              xlabel_fontsize=self.fontsize, ylabel_fontsize=self.fontsize, tight_layout=True,
                              savefig=True, main_title='output_model_' + str(epoch),
                              legend_fontsize=self.fontsize)

    def plot_gradients(self, model, epoch):
        all_grads_tcn = []
        for param in model.network.parameters():
            all_grads_tcn += list(param.grad.view(-1).numpy())
        all_grads_classifier = []
        for param in model.classifier.parameters():
            all_grads_classifier += list(param.grad.view(-1).numpy())

        fig, axes = graph.create_figure(subplots=(1, 1), figsize=(8, 8), sharey=False)

        axes[0][0].hist(all_grads_tcn, label='TCN', bins=50)
        axes[0][0].hist(all_grads_classifier, label='classifier', bins=50)
        plt.yscale('log')

        graph.complete_figure(fig, axes, put_legend=[[True]],
                              xticks_fontsize=self.ticks_fontsize, yticks_fontsize=self.ticks_fontsize,
                              xlabel_fontsize=self.fontsize, ylabel_fontsize=self.fontsize, tight_layout=True,
                              savefig=True, main_title='grads_model_' + str(epoch),
                              legend_fontsize=self.fontsize)

    def run_epoch(self, data_loader, optimizer, model, device, train_flag, epoch_number):
        if train_flag is True:
            add_str = 'Train'
        else:
            add_str = 'Val'

        if train_flag is True:
            model.train()
        else:
            model.eval()

        dict_epoch_loss = {}
        for data in tqdm(data_loader):
            if train_flag is True:
                optimizer.zero_grad()

            signal, label = data
            signal = signal.to(device)
            label = label.to(device)

            e1, e2, out = model(signal)
            dict_loss = self.loss(e1, e2, out, label, model)

            if train_flag is True:
                for key_loss in dict_loss:
                    dict_loss[key_loss].backward()
                optimizer.step()

            for key_loss in dict_loss:
                if key_loss in dict_epoch_loss.keys():
                    dict_epoch_loss[key_loss] += dict_loss[key_loss].data.item()
                else:
                    dict_epoch_loss[key_loss] = dict_loss[key_loss].data.item()

        for key_loss in dict_epoch_loss:
            dict_epoch_loss[key_loss] /= len(data_loader)
            self.writer.add_scalar(key_loss + "/" + add_str, dict_epoch_loss[key_loss], epoch_number)

        running_loss = dict_epoch_loss['mse_loss']
        log = add_str + " Loss: {:.4f}  | ".format(running_loss)
        return running_loss, log

    def save_model(self, best_test_loss, eval_loss, model, epoch, early_stopping=15):
        if best_test_loss > eval_loss:
            torch.save(model.state_dict(), os.path.join(self.saved_path, 'model.pth'))
            best_test_loss = eval_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(self.saved_path, 'model_epoch_' + str(epoch) + '.pth'))

        if self.epochs_without_improvement >= early_stopping:
            stop_training = True
        else:
            stop_training = False

        return best_test_loss, stop_training

    def prepare_data(self, curr_params):
        databases = self.get_databases_osa(curr_params, compute_shhs=True, compute_shhs2=True,
                                           compute_external=False)

        train_dataset = merge_generator(databases.data_train_osa_SHHS, databases.data_train_osa_SHHS2)
        test_dataset = merge_generator(databases.data_test_osa_SHHS, databases.data_test_osa_SHHS2)

        train_dataset.y = np.clip(train_dataset.y, a_min=0, a_max=100)
        test_dataset.y = np.clip(test_dataset.y, a_min=0, a_max=100)

        mean_train = np.mean(train_dataset.X)
        std_train = np.std(train_dataset.X)

        train_OximetryDataset = OximetryDataset(train_dataset.X, train_dataset.y, self.model_name, curr_params,
                                                mean=mean_train, std=std_train)
        train_dataloader = torch.utils.data.DataLoader(train_OximetryDataset, batch_size=curr_params['batch_size'],
                                                       shuffle=True, num_workers=0)

        val_OximetryDataset = OximetryDataset(test_dataset.X, test_dataset.y, self.model_name, curr_params,
                                              mean=mean_train, std=std_train)
        val_dataloader = torch.utils.data.DataLoader(val_OximetryDataset, batch_size=curr_params['batch_size'],
                                                     shuffle=False, num_workers=0)

        return train_dataloader, val_dataloader

    def get_model(self, curr_params, no_features=False):
        if self.model_name == 'duplo':
            return DuploClassifier(1, 513, 1, curr_params).to(self.device)
        elif self.model_name == 'resnet':
            return ResNet1D(in_channels=1, base_filters=curr_params['base_filters'],
                            kernel_size=curr_params['kernel_size'], stride=curr_params['stride'],
                            groups=1, n_block=curr_params['n_block'], verbose=False).to(self.device)
        elif self.model_name == 'TCN':
            num_channels = curr_params['n_blocks'] * [curr_params['n_filter_initial']]
            return TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=curr_params['kernel_size'],
                                   dropout=curr_params['dropout'])
        else:
            raise WrongParameter('self.model_name must be in {duplo, resnet, TCN')

    @staticmethod
    def sample_augmentations():
        aug = ['permutation_augmentation', 'scaling_augmentation', 'rotation_augmentation',
               'magnitude_warp_augmentation', 'time_warp_augmentation', 'data_augmentation', 'jitter_augmentation']

        dict_aug = {}
        for curr_aug in aug:
            dict_aug[curr_aug] = random.choice([True, False])

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
                'stride': [1, 3],
                'n_block': [4, 5, 6, 7, 8, 9, 12]
            }
        elif self.model_name == 'TCN':
            return {
                'n_blocks': [2, 3, 4, 5, 6],
                'n_filter_initial': [2, 4, 8, 16, 32, 64],
                'kernel_size': [3, 5, 7, 9, 11, 13],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        else:
            raise WrongParameter('self.model_name must be in {duplo, resnet, TCN')

    def run(self):
        dict_params_it = self.get_dict_params()

        for i in range(100):
            self.assure_reproducibility(5*i)
            self.writer = SummaryWriter()

            dict_params = sample_from_dict(dict_params_it)
            dict_aug = self.sample_augmentations()
            curr_params = {**self.params_it, **dict_params, **dict_aug}

            train_dataloader, val_dataloader = self.prepare_data(curr_params)
            model = self.get_model(curr_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=curr_params['learning_rate'])

            best_val_loss = np.inf
            for epoch in range(1, self.num_epochs + 1):
                epoch_time = time.time()

                _, train_log = self.run_epoch(train_dataloader, optimizer, model, self.device, train_flag=True,
                                              epoch_number=epoch)

                with torch.no_grad():
                    val_loss, val_log = self.run_epoch(val_dataloader, optimizer, model, self.device, train_flag=False,
                                                       epoch_number=epoch)

                best_val_loss, stop_training = self.save_model(best_val_loss, val_loss, model, epoch)

                epoch_time = time.time() - epoch_time
                log = 'Epoch {:.1f} |'.format(epoch) + train_log + val_log
                log += "Epoch Time: {:.2f} secs | Best loss is {:.2f}".format(epoch_time, best_val_loss)
                print(log)

                if stop_training is True:
                    break

            dict_final = self.inference(curr_params, val_dataloader)
            dict_final['val_loss'] = best_val_loss
            save_dict(dict_final, file_name=os.path.join(self.saved_path, 'results.csv'))

            self.writer.close()

    def inference(self, curr_params, val_dataloader=None):
        if val_dataloader is None:
            _, val_dataloader = self.prepare_data(curr_params)

        model = self.get_model(curr_params)
        model.load_state_dict(torch.load(os.path.join(self.saved_path, 'model.pth')))

        y_pred, y_test = [], []
        for i, data in enumerate(tqdm(val_dataloader)):
            signal, label = data
            _, _, output = model(signal)

            y_pred.append(output)
            y_test.append(label)

        y_pred = torch.cat(y_pred, 0).detach().numpy()
        y_test = torch.cat(y_test, 0)

        dict_results = compute_metrics_osa(y_pred, y_test)
        dict_final = {**curr_params, **dict_results}
        return dict_final


if __name__ == '__main__':
    main_pytorch = Main_Pytorch(short_sample=False, device_ids=0, num_epochs=5, model_name='TCN')
    main_pytorch.run()
