import sys

sys.path.append('/home/jeremy.levy/Jeremy/copd_osa')

import torch
from tqdm import tqdm
import time
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import utils.graphics as graph
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from icecream import ic

from DL.Main import Main
from DL.pytorch_version.Dataloader import OximetryDataset
from DL.pytorch_version.models.ResNet import ResNet1D
from DL.util_funcs import compute_metrics_osa
from utils.utils_func import save_dict
from DL.pytorch_version.models.Duplo import DuploClassifier
from DL.pytorch_version.models.TCN import TemporalConvNet
from utils.help_classes import WrongParameter, Databases_osa
from DL.Dataloader.DataGeneratorFullSignal import merge_generator
from DL.pytorch_version.Custom_Loss import CustomLoss
from DL.pytorch_version.SettingsDL import GetDict
from DL.pytorch_version.warmup import UntunedLinearWarmup


def sample_from_dict(dict_params_it):
    dict_params = {}
    for key in dict_params_it:
        dict_params[key] = random.choice(dict_params_it[key])
    return dict_params


class Main_Pytorch(Main):
    def __init__(self, short_sample, num_epochs, model_name, multi_gpu, device):
        super().__init__(type_run='', model_name=model_name, epoch_1plot=0, epoch_1=0, epoch_2=0,
                         short_sample=short_sample)

        if short_sample is True:
            self.num_epochs = 30
        else:
            self.num_epochs = num_epochs

        self.epochs_without_improvement = 0
        self.multi_gpu = multi_gpu

        self.saved_path = 'data_saved'
        os.makedirs(self.saved_path, exist_ok=True)
        self.ticks_fontsize, self.fontsize, self.letter_fontsize = 15, 15, 15

        self.params_it = {'model_name': model_name,

                          # Learning process
                          'regularization_weight': 0.01, 'learning_rate': 0.0005, 'batch_size': 64,
                          'aug_weight': 0.1, 'supp_weight': 1,

                          # data
                          'sampling': False, 'apply_median_spo2': False, 'all_metadata': 'meta_pobm',
                          'features_ss': False, 'normalization': False, 'rocket_features': False,
                          'double_channel': False, 'loss_osa': 'MSE', 'loss_copd': 'FocalLoss_digit_false',
                          'osa_regression': True, 'full_signal': True,
                          'transform_ahi': None, 'activation_function_copd': None, 'activation_function_osa': None,

                          # STFT
                          'window_stft': 'hamming', 'n_fft': 1024, 'win_length': 128,

                          # Split
                          'train_size_UHV_pc': 100, 'train_size_SHHS_pc': 0.9, 'train_size_WSC_pc': 100,
                          'train_size_CFS_pc': 100, 'train_size_SHHS2_pc': 0.7, 'train_size_numom_pc': 100,
                          }

        if device == 'gpu':
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f"Using {self.device} device")

        self.loss_class = CustomLoss(device=self.device,
                                     regularization_weight=self.params_it['regularization_weight'],
                                     aug_weight=self.params_it['aug_weight'],
                                     supp_weight=self.params_it['supp_weight'])
        self.sample_class = GetDict(self.model_name)

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

    def run_epoch(self, data_loader, optimizer, lr_scheduler, model, train_flag, epoch_number,
                  configuration_run=None):
        if train_flag is True:
            add_str = 'Train'
            model.train()
        else:
            add_str = 'Val'
            model.eval()

        if configuration_run == 'aug_loss':
            aug_loss_flag = True
            sup_loss_flag = False
        elif configuration_run == 'sup_loss':
            aug_loss_flag = False
            sup_loss_flag = True
        else:
            aug_loss_flag = False
            sup_loss_flag = False

        self.loss_class.on_epoch_begin()
        for i, data in enumerate(tqdm(data_loader)):
            if train_flag is True:
                optimizer.zero_grad()

            signal, label, dataset_label = data
            signal = signal.to(self.device)
            label = label.to(self.device)
            dataset_label = dataset_label.to(self.device)

            e1, e2, out = model(signal)
            self.loss_class(e1=e1, e2=e2, y_pred=out, y_target=label, model=model, aug_loss_flag=aug_loss_flag,
                            sup_loss_flag=sup_loss_flag, domain_tag=dataset_label, train_flag=train_flag)

            # if train_flag is True:
            #     optimizer.step()

        self.loss_class.on_epoch_end(len(data_loader), add_str, epoch_number)

        lr_scheduler.step()
        self.loss_class.writer.add_scalar('learning_rate' + "/" + add_str, lr_scheduler.get_last_lr()[0], epoch_number)

        running_loss = self.loss_class.get_running_loss()
        log = add_str + " Loss: {:.4f}  | ".format(running_loss)

        return running_loss, log

    def save_model(self, best_test_loss, eval_loss, model, epoch, i, early_stopping=15):
        if best_test_loss > eval_loss:
            torch.save(model.state_dict(), os.path.join(self.saved_path, 'model_' + str(i) + '.pth'))
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

    def set_database_label(self, databases: Databases_osa):
        databases.data_train_osa_SHHS.set_database_name_label(
            [self.dataset_label_dict['SHHS1']] * databases.data_train_osa_SHHS.X.shape[0])
        databases.data_test_osa_SHHS.set_database_name_label(
            [self.dataset_label_dict['SHHS1']] * databases.data_test_osa_SHHS.X.shape[0])

        databases.data_train_osa_SHHS2.set_database_name_label(
            [self.dataset_label_dict['SHHS2']] * databases.data_train_osa_SHHS2.X.shape[0])
        databases.data_test_osa_SHHS2.set_database_name_label(
            [self.dataset_label_dict['SHHS2']] * databases.data_test_osa_SHHS2.X.shape[0])

        try:
            databases.data_train_osa_UHV.set_database_name_label(
                [self.dataset_label_dict['UHV']] * databases.data_train_osa_UHV.X.shape[0])
            databases.data_test_osa_UHV.set_database_name_label(
                [self.dataset_label_dict['UHV']] * databases.data_test_osa_UHV.X.shape[0])
        except AttributeError:
            pass

        try:
            databases.data_train_osa_CFS.set_database_name_label(
                [self.dataset_label_dict['CFS']] * databases.data_train_osa_CFS.X.shape[0])
            databases.data_test_osa_CFS.set_database_name_label(
                [self.dataset_label_dict['CFS']] * databases.data_test_osa_CFS.X.shape[0])
        except AttributeError:
            pass

        try:
            databases.data_train_osa_WSC.set_database_name_label(
                [self.dataset_label_dict['WSC']] * databases.data_train_osa_WSC.X.shape[0])
            databases.data_test_osa_WSC.set_database_name_label(
                [self.dataset_label_dict['WSC']] * databases.data_test_osa_WSC.X.shape[0])
        except AttributeError:
            pass

        try:
            databases.data_train_osa_numom.set_database_name_label(
                [self.dataset_label_dict['numom']] * databases.data_train_osa_numom.X.shape[0])
            databases.data_test_osa_numom.set_database_name_label(
                [self.dataset_label_dict['numom']] * databases.data_test_osa_numom.X.shape[0])
        except AttributeError:
            pass

    def prepare_data(self, curr_params, configuration_run):
        if (configuration_run == 'sup_loss') or (configuration_run == 'regular_all'):
            compute_external = True
            self.skip_wsc = True
        else:
            compute_external = False

        databases = self.get_databases_osa(curr_params, compute_shhs=True, compute_shhs2=True,
                                           compute_external=compute_external)
        self.set_database_label(databases)

        if (configuration_run is None) or (configuration_run == 'aug_loss') or (configuration_run == 'DSU'):
            train_dataset = merge_generator(databases.data_train_osa_SHHS, databases.data_train_osa_SHHS2)
            test_dataset = merge_generator(databases.data_test_osa_SHHS, databases.data_test_osa_SHHS2)

        elif (configuration_run == 'sup_loss') or (configuration_run == 'regular_all'):
            train_dataset = self.fusion_datasets([databases.data_train_osa_SHHS, databases.data_train_osa_SHHS2,
                                                  databases.data_train_osa_UHV, databases.data_train_osa_CFS])
            test_dataset = self.fusion_datasets([databases.data_test_osa_SHHS, databases.data_test_osa_SHHS2,
                                                 databases.data_test_osa_UHV, databases.data_test_osa_CFS])
        else:
            raise WrongParameter('configuration_run must be in {None, aug_loss, sup_loss}')

        self.mean_train = np.mean(train_dataset.X)
        self.std_train = np.std(train_dataset.X)

        train_dataloader, val_dataloader = self.get_train_val_dataloader(train_dataset, test_dataset, curr_params)
        return train_dataloader, val_dataloader

    def get_train_val_dataloader(self, train_dataset, test_dataset, curr_params):
        train_OximetryDataset = OximetryDataset(train_dataset.X, train_dataset.y, self.model_name, curr_params,
                                                mean=self.mean_train, std=self.std_train,
                                                dataset_label=train_dataset.database_name_label)
        train_dataloader = torch.utils.data.DataLoader(train_OximetryDataset, batch_size=curr_params['batch_size'],
                                                       shuffle=True, num_workers=0)

        val_OximetryDataset = OximetryDataset(test_dataset.X, test_dataset.y, self.model_name, curr_params,
                                              mean=self.mean_train, std=self.std_train,
                                              dataset_label=test_dataset.database_name_label)
        val_dataloader = torch.utils.data.DataLoader(val_OximetryDataset, batch_size=curr_params['batch_size'],
                                                     shuffle=False, num_workers=0)

        return train_dataloader, val_dataloader

    def prepare_external_data(self, curr_params, add_wsc=True):
        """
        Return list of tuples: (Train dataloader, Test dataloader and name of the dataset.
        For each external databases.
        """
        if add_wsc is True:
            self.skip_wsc = False
        else:
            self.skip_wsc = True

        databases = self.get_databases_osa(curr_params, compute_shhs=True, compute_shhs2=True, compute_external=True)
        self.set_database_label(databases)

        list_train_dataset = [databases.data_train_osa_SHHS, databases.data_train_osa_SHHS2,
                              databases.data_train_osa_UHV, databases.data_train_osa_CFS]
        list_test_dataset = [databases.data_test_osa_SHHS, databases.data_test_osa_SHHS2, databases.data_test_osa_UHV,
                             databases.data_test_osa_CFS]
        list_name_dataset = ['SHHS1', 'SHHS2', 'UHV', 'CFS']

        if add_wsc is True:
            list_train_dataset.append(databases.data_train_osa_WSC)
            list_test_dataset.append(databases.data_test_osa_WSC)
            list_name_dataset.append('WSC')

        all_data_list = []
        for train_data, test_data, name_data in zip(list_train_dataset, list_test_dataset, list_name_dataset):
            train_dataloader, val_dataloader = self.get_train_val_dataloader(train_data, test_data, curr_params)
            all_data_list.append((train_dataloader, val_dataloader, name_data))

        return all_data_list

    def get_model(self, curr_params, configuration_run=None):
        if self.model_name == 'duplo':
            model = DuploClassifier(1, 513, 1, curr_params).to(self.device)
        elif self.model_name == 'resnet':
            model = ResNet1D(in_channels=1, base_filters=curr_params['base_filters'],
                             kernel_size=curr_params['kernel_size'], stride=curr_params['stride'],
                             groups=1, n_block=curr_params['n_block'], verbose=False,
                             dropout=curr_params['dropout'], configuration_run=configuration_run).to(self.device)
        elif self.model_name == 'TCN':
            model = TemporalConvNet(curr_params, configuration_run=configuration_run).to(self.device)
        else:
            raise WrongParameter('self.model_name must be in {duplo, resnet, TCN')

        if self.multi_gpu is True:
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        return model

    def train_model(self, train_dataloader, val_dataloader, optimizer, lr_scheduler, model, i,
                    configuration_run=None):
        best_val_loss = np.inf
        for epoch in range(1, self.num_epochs + 1):
            epoch_time = time.time()

            try:
                _, train_log = self.run_epoch(train_dataloader, optimizer, lr_scheduler, model,
                                              train_flag=True, epoch_number=epoch, configuration_run=configuration_run)
            except KeyboardInterrupt:
                break

            with torch.no_grad():
                val_loss, val_log = self.run_epoch(val_dataloader, optimizer, lr_scheduler, model,
                                                   train_flag=False, epoch_number=epoch)

            best_val_loss, stop_training = self.save_model(best_val_loss, val_loss, model, epoch, i)

            epoch_time = time.time() - epoch_time
            log = 'Epoch {:.1f} |'.format(epoch) + train_log + val_log
            log += "Epoch Time: {:.2f} secs | Best loss is {:.2f}".format(epoch_time, best_val_loss)
            print(log)

            if stop_training is True:
                break
        return best_val_loss

    def run(self):
        dict_params_it = self.sample_class.get_dict_params()
        seed_array = np.random.randint(low=0, high=100000, size=50000)

        for i in range(100):
            self.assure_reproducibility(seed_array[i])
            self.writer = SummaryWriter()

            dict_params = sample_from_dict(dict_params_it)
            dict_aug = self.sample_class.sample_augmentations()
            dict_data = self.sample_class.sample_data_params()
            curr_params = {**self.params_it, **dict_params, **dict_aug, **dict_data}

            print(curr_params)
            train_dataloader, val_dataloader = self.prepare_data(curr_params, configuration_run=None)
            model = self.get_model(curr_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=curr_params['learning_rate'])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)

            best_val_loss = self.train_model(train_dataloader, val_dataloader, optimizer,
                                             lr_scheduler, model, i)

            dict_results = self.inference(curr_params, model_idx=i, val_dataloader=val_dataloader)
            dict_final = {**curr_params, **dict_results, 'val_loss': best_val_loss, 'seed': seed_array[i]}

            save_dict(dict_final, file_name=os.path.join(self.saved_path, 'results_mae.csv'))

            self.writer.close()

    def inference(self, curr_params, model_idx, val_dataloader, model=None, add_str=''):
        if model is None:
            model = self.get_model(curr_params)
            model.load_state_dict(torch.load(os.path.join(self.saved_path, 'model_' + str(model_idx) + '.pth')))

        y_pred, y_test = [], []
        for i, data in enumerate(tqdm(val_dataloader, desc=add_str)):
            signal, label, _ = data
            signal = signal.to(self.device)
            label = label.to(self.device)

            _, _, output = model(signal)

            y_pred.append(output)
            y_test.append(label)

        y_pred = torch.cat(y_pred, 0).detach().cpu().numpy()
        y_test = torch.cat(y_test, 0).cpu().numpy()

        dict_results = compute_metrics_osa(y_pred, y_test)
        dict_return = {
            add_str + '_r_square': dict_results['R_square_osa'],
            add_str + '_ICC': dict_results['ICC'],
            add_str + '_f1': dict_results['f1_osa'],
        }

        return dict_return

    def generalization_run(self, filename, configuration_run, log_dir):
        dict_run = self.sample_class.retrieve_dict_run(filename)
        print(dict_run)

        self.assure_reproducibility(dict_run['seed'])

        train_dataloader, val_dataloader = self.prepare_data(dict_run, configuration_run=configuration_run)

        model = self.get_model(dict_run, configuration_run=configuration_run)
        if configuration_run == 'aug_loss':
            model.load_state_dict(torch.load(os.path.join(self.saved_path, 'model_regular_34_315.pth')))
        elif configuration_run == 'sup_loss':
            print('loading pre-trained model')
            model.load_state_dict(torch.load(os.path.join(self.saved_path, 'model_regular_34_315.pth')))

        optimizer = torch.optim.Adam(model.parameters(), lr=dict_run['learning_rate'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.3)

        self.loss_class.set_writer(log_dir=log_dir)

        best_val_loss = self.train_model(train_dataloader, val_dataloader, optimizer, lr_scheduler,
                                         model, i=100, configuration_run=configuration_run)
        dict_run['val_loss'] = best_val_loss

        external_data = self.prepare_external_data(curr_params=dict_run, add_wsc=False)
        for train_data, val_data, name_data in external_data:
            dict_res = self.inference(dict_run, model_idx=100, val_dataloader=val_data, model=model, add_str=name_data)
            dict_run = {**dict_run, **dict_res}

        self.loss_class.writer.close()
        save_dict(dict_run, file_name=os.path.join(self.saved_path, 'generalization_run_' + configuration_run + '.csv'))


if __name__ == '__main__':
    main_pytorch = Main_Pytorch(short_sample=False, num_epochs=200, model_name='resnet', multi_gpu=False, device='cpu')
    # main_pytorch.run()

    # main_pytorch.generalization_run(filename='config_resnet.csv', configuration_run='regular')
    # main_pytorch.generalization_run(filename='config_resnet.csv', configuration_run='aug_loss')
    main_pytorch.generalization_run(filename='config_resnet.csv', configuration_run='sup_loss', log_dir='sup_loss')
    # main_pytorch.generalization_run(filename='config_resnet.csv', configuration_run='DSU')
    # main_pytorch.generalization_run(filename='config_resnet.csv', configuration_run='regular_all', log_dir='scheduling')
