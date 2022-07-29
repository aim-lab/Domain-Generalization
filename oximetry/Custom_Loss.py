from torch import nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class CustomLoss(nn.Module):
    def __init__(self, device, regularization_weight=0.0001, aug_weight=0.1, supp_weight=0.001):
        super(CustomLoss, self).__init__()

        self.regularization_weight = regularization_weight
        self.supp_weight = supp_weight
        self.aug_weight = aug_weight

        self.device = device
        self.regression_loss = nn.L1Loss()

        self.dict_epoch_loss = {}
        self.writer = None

    def forward(self, e1, e2, y_pred, y_target, model, train_flag, aug_loss_flag=False, sup_loss_flag=False,
                domain_tag=None):
        """

        @param e1:  Encoding of the original data, with encoder e1
        @param e2:  Encoding of the original data, with encoder e2
        @param y_pred:  Predictions of the model
        @param y_target:    GT
        @param model:   The model, with e2_model as attribute (to compute phi_aug)
        @param aug_loss_flag:   Boolean, whether to compute aug_loss
        @param sup_loss_flag:  Boolean, whether to compute supp_loss
        @param domain_tag:  Domain for each sample
        @return:    Dictionary of all loss
        """
        mse_loss = self.regression_loss(y_pred, y_target)
        loss = mse_loss

        l2_reg = torch.tensor(0., device=self.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        reg_loss = self.regularization_weight * l2_reg
        loss += reg_loss

        dict_loss = {
            'mse_loss': mse_loss,
            'reg_loss': reg_loss
        }

        if aug_loss_flag is True:
            aug_e1 = self.create_aug(e1)
            phi_aug = model.e2_model(aug_e1)
            aug_loss = self.L_aug(e2, phi_aug)

            aug_loss_value = self.aug_weight * aug_loss
            loss += aug_loss_value
            dict_loss['aug_loss'] = aug_loss_value

        if sup_loss_flag is True:
            assert domain_tag is not None
            supp_loss = self.L_supp(e2, domain_tag)

            supp_loss_value = self.supp_weight * supp_loss
            loss += supp_loss_value
            dict_loss['supp_loss'] = supp_loss_value

        if train_flag is True:
            loss.backward(retain_graph=True)

        self.update_dict_loss(dict_loss)
        return loss

    @staticmethod
    def create_aug(V, alpha=0.9):
        """
        This function creates a positive augmentation for every representation in the mini-batch according to mixup
        algorithm.
        :param V: a mini-batch of embeddings (after e1).
        :param alpha: a fraction that controls the "positivity" of the augmentaiton. The closer it is to 1, the more
        likely the embedding to resemble the representation.
        :return: matrix of augmentations in the same size of the mini-batch.
        """
        vec_idx = torch.arange(V.shape[0])
        A = torch.zeros_like(V)
        for j, v in enumerate(V):  # runs over the first dimension which is the number of examples per batch
            lmbda = (1 - alpha) * torch.rand(1).item() + alpha  # setting uniformly distributed r.v in range [alph,1]
            vec_neg = vec_idx[~np.isin(vec_idx, j)]
            perm = torch.randperm(len(vec_neg))
            v_bar = V[perm[0]]
            A[j] = lmbda * v + (1 - lmbda) * v_bar
        return A

    @staticmethod
    def L_aug(Z, phi_A, n=100, eps=10.0 ** -6, tau=10.0 ** -6):
        """
        This function calculates the mutual information (MI) ratio between the representation to its' positive augmentation
        w.r.t. the MI of the representation to its' n negative augmentations.
        :param Z: mini-batch of embeddings (after e1+e2).
        :param phi_A: mini-batch of augmentations' embeddings (after e2).
        :param n: number of negative augmentations per example (must be lower than batch size).
        :param eps:
        :param tau:
        :return: mean MI ratio across the mini-batch.
        """
        vec_idx = torch.arange(Z.shape[0])
        I_Z_A = torch.Tensor([0.0])
        if n >= Z.shape[0]:
            n = Z.shape[0] - 1
        for j, pos_pair in enumerate(zip(Z, phi_A), 0):
            z, phi_A_pos = pos_pair
            vec_neg = vec_idx[~np.isin(vec_idx, j)]
            perm = torch.randperm(len(vec_neg))
            v_bar = phi_A[perm[:n]]
            # set v_bar as a matrix with n rows and flattened data
            A = torch.cat((phi_A_pos.flatten().unsqueeze(0), v_bar.view(v_bar.size(0), -1)))
            sim = torch.exp(tau * torch.matmul(A, z.flatten()))
            L = torch.log(sim[0] / (eps + torch.sum(sim)))

            # can happen if tau is not enough to lower the exp in sim
            if not (torch.isnan(L) or torch.isinf(L)):
                I_Z_A -= L

        mean_I_Z_A = I_Z_A / len(Z)
        return mean_I_Z_A

    @staticmethod
    def L_supp(Z, domain_tag, eps=10.0 ** -6, tau=10.0 ** -10):
        """
        This function calculates the support loss to minimize domain-specific effects.
        :param Z: mini-batch of embeddings (after e1+e2).
        :param domain_tag: domain labels (age, hospital number etc.).
        :param eps:
        :param tau:
        :return: support loss as in the paper.
        """

        B_Z_D = torch.Tensor([0.0])

        for j, z_domain_pair in enumerate(zip(Z, domain_tag), 0):
            z, domain = z_domain_pair
            Z_D = Z[domain_tag != domain]
            # todo: should we drop z from Z?
            if Z_D.size()[0] != 0:
                nom = torch.sum(torch.exp(tau * torch.matmul(Z_D.view(Z_D.size(0), -1), z.flatten())))
                den = torch.sum(torch.exp(tau * torch.matmul(Z.view(Z.size(0), -1), z.flatten())))
                L = torch.log(nom / (den + eps))

                if not (torch.isnan(L) or torch.isinf(L)):
                    B_Z_D -= L

        mean_B_Z_D = B_Z_D / len(Z)
        return mean_B_Z_D[0]

    def on_epoch_begin(self):
        self.dict_epoch_loss = {}

    def update_dict_loss(self, dict_loss):
        for key_loss in dict_loss:
            if key_loss in self.dict_epoch_loss.keys():
                self.dict_epoch_loss[key_loss] += dict_loss[key_loss].data.item()
            else:
                self.dict_epoch_loss[key_loss] = dict_loss[key_loss].data.item()

    def on_epoch_end(self, len_dataloader, add_str, epoch_number):
        for key_loss in self.dict_epoch_loss:
            self.dict_epoch_loss[key_loss] /= len_dataloader
            self.writer.add_scalar(key_loss + "/" + add_str, self.dict_epoch_loss[key_loss], epoch_number)

    def get_running_loss(self):
        return self.dict_epoch_loss['mse_loss']

    def set_writer(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
