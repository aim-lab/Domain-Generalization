import os
import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, losses
from data_loader import TorchDataset
import time
from sklearn.metrics import classification_report
from run import Run
from sklearn.model_selection import train_test_split
from extra_functions import BuildCNN


def anc_pos_neg(i, out1, out2, lbl1, lbl2):
    anchor = out1[i, :]
    flag = 1
    pos = 1
    p_idx = torch.where(lbl2 == lbl1[i])[0]
    if p_idx.shape[0] > 1:
        p_idx = p_idx[0].item()
    elif p_idx.shape[0] == 0:
        flag = 0
        pos = 0
    else:
        p_idx = p_idx.item()
    n_idx = torch.where(lbl2 != lbl1[i])[0]
    if n_idx.shape[0] > 1:
        n_idx = n_idx[torch.randint(n_idx.shape[0], (1,)).item()].item()
    else:
        n_idx = n_idx.item()
    if flag != 0:
        pos = out2[p_idx, :]
    neg = out2[n_idx, :]
    return anchor, pos, neg


def id_triplet_loss(out1, out2, lbl1, lbl2, lmbda=0.5):
    cos = nn.CosineSimilarity()
    soft = nn.Softmax()
    eps = 0.0000005
    l = 0
    for i in range(out1.shape[0]):
        anchor, pos, neg = anc_pos_neg(i, out1, out2, lbl1, lbl2)
        cov_anc = cov(anchor, anchor, rowvar=True)
        cov_anc_neg = cov(anchor, neg, rowvar=True)
        cov_neg = cov(neg, neg, rowvar=True)
        M = cov_anc - cov_anc_neg.matmul(torch.inverse(cov_neg).t()).matmul(cov_anc_neg.t())
        I = -0.5*(1/M.shape[0])*torch.log(M + eps).diag().nansum()
        if I.isnan():
            I = 0
        Lce = soft(anchor).matmul(soft(neg))  # .mean(-torch.sum(anchor * torch.log(neg), 1))
        if Lce.isnan():
            Lce = 0
        if type(pos) == int:
            L_cosine = 0
        else:
            L_cosine = 1 - cos(anchor.unsqueeze(dim=0), pos.unsqueeze(dim=0))
            if L_cosine.isnan():
                L_cosine = 0
        l += L_cosine + lmbda*Lce + (1 - lmbda)*I
    loss = l/out1.shape[0]
    return loss


def cosine_crossentropy_loss(out1, out2, lbl1, lbl2, lmbda=0.5):
    cos = nn.CosineSimilarity()
    # res1 = torch.abs(cos(out1, out2))  # , torch.abs(cos(out1, out2))
    # res2 = torch.abs(1 - cos(out1, out2))
    m = nn.ReLU()
    res1 = m(cos(out1, out2))
    res2 = 1 - res1
    res = torch.cat((torch.unsqueeze(res1, 0), torch.unsqueeze(res2, 0)), 0)
    res = res.t()
    lbl = 1*(lbl1 == lbl2)
    nll_loss = nn.CrossEntropyLoss()
    L1_loss = nn.L1Loss()
    loss = nll_loss(res, lbl) + lmbda*L1_loss(res1, res2)
    # loss = cross_entropy(torch.tensor(res2, dtype=float, requires_grad=True), torch.tensor(lbl, dtype=torch.double))
    # lbl = torch.tensor([1*indicator, 1-1*indicator])
    return loss


def cov(m1, m2, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    ########################ORIGINAL######################################
    # if m.dim() > 2:
    #     raise ValueError('m has more than 2 dimensions')
    # if m.dim() < 2:
    #     m = m.view(1, -1)
    # m = m1
    # if not rowvar and m.size(0) != 1:
    #     m = m.t()
    # # m = m.type(torch.double)  # uncomment this line if desired
    # fact = 1.0 / (m.size(1) - 1)
    # m -= torch.mean(m, dim=1, keepdim=True)
    # mt = m.t()  # if complex: mt = m.t().conj()
    # return fact * m.matmul(mt).squeeze()
    ######################TWO INPUTS###########################
    if m1.dim() < 2:
        m1 = m1.view(1, -1)
        m2 = m2.view(1, -1)
    # if not rowvar and m1.size(0) != 1:
        # m1 = m1.t()
        # m2 = m2.t()
    # m = m.type(torch.double)  # uncomment this line if desired

    fact = 1.0 / (m1.size(1) - 1)
    m1 = m1 - torch.mean(m1, dim=1, keepdim=True)
    m2 = m2 - torch.mean(m2, dim=1, keepdim=True)
    mt = m1.t()  # if complex: mt = m.t().conj()
    return fact * mt.matmul(m2).squeeze()


def cosine_loss(out1, out2, lbl1, lbl2, flag=0, lmbda=1, b=0):
    """

    :param out1: representation in latent space of input1
    :param out2: representation in latent space of input2
    :param lbl1: id of input1
    :param lbl2: id of input2
    :param flag: 0: return only the loss, 1: return cosine distance and equality label
    :param lmbda: controls FAR. The larger lmbda is, the smaller the FAR
    :param b: scalar in [-1,1]. Controls the threshold for deciding if two inputs are the same or not. By default, if out1 and out2 are
     "less" than perpendicular thus they considered the same. We can treat b as cos(theta) where theta is the angle
     between representations.
    :return:
    """
    if flag == 2:
        cos = nn.MSELoss()
    else:
        cos = nn.CosineSimilarity()
    res = cos(out1, out2)
    res = res.t()
    y = 1*(lbl1 == lbl2)
    # batch_loss = lmbda1*(1 - y) * res + lmbda2*y*(1 - res)
    #
    # batch_loss = (1-y*(1+lmbda))*res
    if flag == 2:
        batch_loss = (1-2*y)*res
        return batch_loss.mean()
    else:
        batch_loss = -lmbda*y*(b + res) + (1 - y)*res
    loss = batch_loss.mean()
    if flag:
        return res, y
    else:
        return loss


def MI_cosine_loss(out1, out2, lbl1, lbl2, flag=0):
    eps = 0.0000005
    loss = 0
    for i in range(out1.shape[0]):
        for j in range(i, out2.shape[0]):  # Notice to start from i to reduce redundency
            Y = out1[i, :]
            P = out2[j, :]
            cov_y = cov(Y, Y, rowvar=True)
            cov_yp = cov(Y, P, rowvar=True)
            cov_p = cov(P, P, rowvar=True)
            M1 = cov_y - cov_yp.matmul(torch.inverse(cov_p).t()).matmul(cov_yp.t())
            M2 = cov_p - cov_yp.t().matmul(torch.inverse(cov_y).t()).matmul(cov_yp)
            M = M1 + M2
            # I = -0.5*torch.log(eps + torch.det(M))
            I = -0.5*(1/Y.shape[0])*torch.log(M).diag().nansum()
            y = 1*(lbl1[i] == lbl2[j])
            l_cosine = cosine_loss(Y.unsqueeze(0), P.unsqueeze(0), lbl1[i], lbl2[j])
            loss += y*(1 - l_cosine) + (1 - y)*I
    loss /= (out1.shape[0] * out2.shape[0])
    # Y = cov(out1, out1, rowvar=True)
    # P = cov(out2, out2, rowvar=True)
    # YP = cov(out1, out2, rowvar=True)
    # PY = YP.t()
    # inv_P_tr = torch.inverse(P).t()
    # inv_Y_tr = torch.inverse(Y).t()
    # M1 = Y - YP.matmul(inv_P_tr).matmul(PY)
    # M2 = P - PY.matmul(inv_Y_tr).matmul(YP)
    # M = M1 + M2
    # eps = 0.0000005
    # I_YP = -0.5*torch.log(torch.diag(M))
    # res, y = cosine_loss(out1, out2, lbl1, lbl2, flag=1)
    # # r = y.sum()
    # # batch_loss = r*y*(1 - res) + (y.shape[0]-r)*(1-y)*I_YP
    # batch_loss = y*(1 - res) + (1-y)*I_YP
    # loss = batch_loss.mean()
    if flag:
        pass
        # return batch_loss, y
    else:
        return loss


def likelihood_loss():
    pass


def kernel_loss():
    pass


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=2.0, mode='normal'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.mode = mode

    # def check_type_forward(self, in_types):
    #     assert len(in_types) == 3
    #
    #     x0_type, x1_type, y_type = in_types
    #     assert x0_type.size() == x1_type.shape
    #     assert x1_type.size()[0] == y_type.shape[0]
    #     assert x1_type.size()[0] > 0
    #     assert x0_type.dim() == 2
    #     assert x1_type.dim() == 2
    #     assert y_type.dim() == 1

    def forward(self, out1, out2, lbl1, lbl2, flag=0):
        ############### normal constructive#######################3
        # self.check_type_forward((x0, x1, y))
        if self.mode == 'normal':
            # euclidian distance
            diff = out1 - out2
            dist_sq = torch.sum(torch.pow(diff, 2), 1)
            dist = torch.sqrt(dist_sq)

            mdist = self.margin - dist
            dist = torch.clamp(mdist, min=0.0)
            y = 1*(lbl1 == lbl2)
            loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
            loss = torch.sum(loss) / 2.0 / out1.size()[0]
        else:
            cos = nn.CosineSimilarity()
            h = 0
            l_ij = 0
            none_count = 0
            for i in range(out1.shape[0]):
                anchor = out1[i, :]
                temp = cos(anchor.tile((out1.shape[0], 1)), out2)
                # for j in range(i, out2.shape[0]):
                flag_none = 1
                p_idx = torch.where(lbl2 == lbl1[i])[0]
                if p_idx.shape[0] > 1:
                    p_idx = p_idx[0].item()
                elif p_idx.shape[0] == 0:
                    flag_none = 0
                    none_count += 1
                    p_idx = 0
                else:
                    p_idx = p_idx.item()
                n_idx = torch.where(lbl2 != lbl1[i])[0]
                if n_idx.shape[0] > 1:
                    n_idx = n_idx[torch.randint(n_idx.shape[0], (1,)).item()].item()
                else:
                    n_idx = n_idx.item()
                    # y = 1*(lbl1 == lbl2)
                    # y = 1*(lbl1[i] == lbl2[j])
                    # nom = torch.exp(temp[j])
                    # den = torch.exp(temp).sum()
                    # h += 1
                    # d_ij = -torch.log(nom / (den - nom))
                    # l_ij += y*d_ij + (1-y)*torch.clamp(self.margin - d_ij, min=0.0)
                l_ij += torch.clamp(self.margin - (flag_none*temp[p_idx] - temp[n_idx]), min=0.0)
            loss = l_ij / out1.shape[0]
            # res = cos(out1, out2)
            # dist_sq = res.t()
            # y = 1*(lbl1 == lbl2)
            # batch_loss = 2*y*(1 - dist_sq) + (1-y)*torch.pow(1 + dist_sq, 2)
            # batch_loss = nn.Softmax(dist_sq)
            # loss = batch_loss.mean()
        if flag:
            return dist_sq, y
        else:
            return loss


class DeepModels():
    def __init__(self, data, n_nets, device):
        self.data_loader = data
        self.ds_train = None
        self.ds_test = None
        self.ds_train_val = None
        self.ds_test_val = None
        self.n_nets = n_nets
        self.chosen_model = None
        self.model = None
        self.model2 = None
        self.optimizer = None
        self.optimizer2 = None
        self.device = device
        self.batch_size_train = None
        self.batch_size_test = None

    def choose_model(self, model_name, label='id', mode='train_test', **kwargs):
        self.ds_train = TorchDataset(self.data_loader, label, 'train')
        self.ds_test = TorchDataset(self.data_loader, label, 'test')
        # dataset_size = len(self.ds_train)
        # validation_split = .2
        # random_seed = 42
        # indices = list(range(dataset_size))
        # split = int(np.floor(validation_split * dataset_size))
        # train_indices, val_indices = indices[split:], indices[:split]
        # val_dataset = Subset(self.ds_train, val_indices)
        if mode != 'train_test':
            X_train, X_test, y_train, y_test = train_test_split(torch.transpose(self.ds_train.X, 0, 1), self.ds_train.y, test_size=0.33,
                                                                stratify=self.ds_train.y, random_state=42)
            self.ds_train = TorchDataset((X_train, y_train), label, mode='train_val')
            self.ds_test = TorchDataset((X_test, y_test), label, mode='test_val')

        if model_name == 'AE':
            self.chosen_model = AE()
        if model_name == 'CNN':
            if label == 'id':
                self.chosen_model = BuildCNN(self.data_loader.dataset_name, num_labels=len(self.ds_train.hash_id), **kwargs)
            else:
                self.chosen_model = BuildCNN(self.data_loader.dataset_name, num_labels=len(np.unique(self.ds_train.y)), **kwargs)
        if model_name == 'TruncCNNtest':
            self.chosen_model = TruncCNNtest(list(self.model.children()), **kwargs)
        if model_name == 'AdverserailCNN':
            self.chosen_model = AdverserailCNN(self.data_loader.dataset_name, **kwargs)
        if model_name == 'Advrtset':
            self.chosen_model = Advrtset(self.data_loader.dataset_name, **kwargs)

    def set_model(self, lr=1e-3, optimizer=torch.optim.Adam, **kwargs):
        # self.device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # self.ds_train = TorchDataset(self.data_loader, 'train')
        self.batch_size_train = int(np.ceil(0.15*self.ds_train.X.shape[1]))
        if self.batch_size_train <= 15:
            self.batch_size_train = int(np.ceil(0.5*self.ds_train.X.shape[1]))
        # self.ds_test = TorchDataset(self.data_loader, 'test')
        self.batch_size_test = int(np.ceil(0.15*self.ds_test.X.shape[1]))
        if self.n_nets == 1:
            self.model = self.chosen_model.to(self.device)
            kldiv = nn.KLDivLoss()
            self.optimizer = optimizer(self.model.parameters(), lr=lr, **kwargs)
        if self.n_nets == 2:
            self.model = self.chosen_model.to(self.device)
            self.model2 = self.chosen_model.to(self.device)
            kldiv = nn.KLDivLoss()
            self.optimizer = optimizer(self.model.parameters(), lr=lr, **kwargs)
            self.optimizer2 = optimizer(self.model.parameters(), lr=lr, **kwargs)

    def train(self, loss_func, epochs=20, n_nets=1):
        if (type(self.chosen_model).__name__ in ['TruncatedCNN', 'AdverserailCNN', 'Advrtset', 'TruncCNNtest']) | (n_nets==2):
            self.model2 = self.model
            self.optimizer2 = self.optimizer
            self.n_nets = 2

        if self.n_nets == 1:
            loss_train = []
            loss_val = []
            acc_train = []
            acc_val = []
            for epoch in range(1, epochs + 1):
                self.model.train()  # put in training mode
                running_loss = 0.0
                epoch_time = time.time()
                trainloader = torch.utils.data.DataLoader(
                    self.ds_train, batch_size=self.batch_size_train, shuffle=True)
                testloader = torch.utils.data.DataLoader(
                    self.ds_test, batch_size=self.batch_size_test, shuffle=False)

                for i, data in enumerate(trainloader, 0):
                    # get the inputs
                    inputs, labels = data
                    # send them to device
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # forward + backward + optimize
                    outputs = self.model(inputs)  # forward pass
                    # outputs = torch.tensor(outputs, dtype=torch.long, requires_grad=True)
                    loss = loss_func(outputs, labels)  # change
                    # always the same 3 steps
                    self.optimizer.zero_grad()  # zero the parameter gradients
                    loss.backward(retain_graph=True)  # backpropagation
                    self.optimizer.step()  # update parameters

                    # print statistics
                    running_loss += loss.data.item()

                # Normalizing the loss by the total number of train batches
                running_loss /= len(trainloader)

                #############VALIDATION#######################
                self.model.eval()  # put in evaluation mode
                total_correct = 0
                total_rr = 0
                val_loss = 0.0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(images)
                        soft_max = nn.Softmax(dim=1)
                        predicted = torch.argmax(soft_max(outputs.data), 1)
                        # sig = nn.Sigmoid() # change
                        # predicted = sig(outputs.data) # change
                        # predicted[predicted >= 0.5] = 1 # change
                        # predicted[predicted < 0.5] = 0 # change
                        total_rr += labels.size(0)
                        total_correct += (predicted == labels).sum().item()
                        val_loss += loss_func(outputs, labels).data.item() # change
                val_loss /= len(testloader)
                # Calculate training/test set accuracy of the existing model
                train_accuracy, _ = self.calculate_accuracy(trainloader)
                test_accuracy, _ = self.calculate_accuracy(testloader)

                loss_train.append(running_loss)
                loss_val.append(val_loss)
                acc_train.append(train_accuracy)
                acc_val.append(test_accuracy)

                log = "Epoch: {} | Training Loss: {:.4f} | Testing Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch, running_loss, val_loss, train_accuracy, test_accuracy)
                # log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy)
                epoch_time = time.time() - epoch_time
                log += "Epoch Time: {:.2f} secs".format(epoch_time)
                print(log)
                # batch_acc.append(train_accuracy)

                # save model
                if epoch % 20 == 0:
                    print('==> Saving model ...')
                    state = {
                        'net': self.model.state_dict(),
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoints'):
                        os.mkdir('checkpoints')
                    torch.save(state, './checkpoints/cifar_cnn_ckpt.pth')

            print('==> Finished Training ...')
            return loss_train, loss_val, acc_train, acc_val


        if self.n_nets == 2:
            for epoch in range(1, epochs + 1):
                self.model.train()  # put in training mode
                self.model2.train()
                running_loss = 0.0
                epoch_time = time.time()
                trainloader1 = torch.utils.data.DataLoader(
                    self.ds_train, batch_size=self.batch_size_train, shuffle=True)
                trainloader2 = torch.utils.data.DataLoader(
                    self.ds_train, batch_size=self.batch_size_train, shuffle=False)
                # trainloader3 = torch.utils.data.DataLoader(self.ds_train, batch_size=len(self.ds_train))
                for i, data in enumerate(zip(trainloader1, trainloader2), 0):
                    # get the inputs
                    inputs1, labels1 = data[0]
                    inputs2, labels2 = data[1]
                    # send them to device
                    inputs1 = inputs1.to(self.device)
                    labels1 = labels1.to(self.device)
                    inputs2 = inputs2.to(self.device)
                    labels2 = labels2.to(self.device)

                    # forward + backward + optimize
                    outputs1 = self.model(inputs1)  # forward pass
                    outputs2 = self.model2(inputs2)
                    if loss_func == 'cosine_crossentropy_loss':
                        loss = cosine_crossentropy_loss(outputs1, outputs2, labels1, labels2, lmbda=0)
                    elif loss_func == 'cosine_loss':
                        loss = cosine_loss(outputs1, outputs2, labels1, labels2)
                    elif loss_func == 'constructive_normal':
                        loss_obj = ContrastiveLoss()
                        loss = loss_obj.forward(outputs1, outputs2, labels1, labels2)
                    elif loss_func == 'constructive_cosine':
                        loss_obj = ContrastiveLoss(mode='cosine')
                        loss = loss_obj.forward(outputs1, outputs2, labels1, labels2)
                    elif loss_func == 'constructive_plus_cosine':
                        loss_obj = ContrastiveLoss(mode='cosine')
                        loss = loss_obj.forward(outputs1, outputs2, labels1, labels2) + cosine_loss(outputs1, outputs2, labels1, labels2)
                    elif loss_func == 'MI_cosine_loss':
                        loss = MI_cosine_loss(outputs1, outputs2, labels1, labels2)
                    elif loss_func == 'id_triplet_loss':
                        loss = id_triplet_loss(outputs1, outputs2, labels1, labels2)
                    else:
                        loss = loss_func
                    # pp = 1*(labels1 == labels2)
                    # if torch.sum(pp) == 0:
                    #     print('No matching labels')
                    # else:
                    #     print('ratio of matching labels is: {:.3f} %'.format((100/len(pp))*torch.sum(pp)))
                    # loss = kldiv(outputs1, outputs2) + kldiv(outputs2, outputs1) # calculate the loss
                    # always the same 3 steps
                    self.optimizer.zero_grad()  # zero the parameter gradients
                    self.optimizer2.zero_grad()
                    # torch.autograd.set_detect_anomaly(True)
                    loss.backward(retain_graph=True)  # backpropagation
                    self.optimizer.step()  # update parameters
                    self.optimizer2.step()

                    # print statistics
                    running_loss += loss.data.item()

                # Normalizing the loss by the total number of train batches
                running_loss /= len(trainloader1)

                # Calculate training/test set accuracy of the existing model
                # train_accuracy, _ = calculate_accuracy(model, trainloader1, device)
                # test_accuracy, _ = calculate_accuracy(model, testloader, device)

                # log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy, test_accuracy)
                log = "Epoch: {} | Loss: {:.4f}  | ".format(epoch, running_loss)
                epoch_time = time.time() - epoch_time
                log += "Epoch Time: {:.2f} secs".format(epoch_time)
                print(log)
                # print("Averaged gradients in first layer of model and model 2 are {:.3f} and {:.3f}".format(torch.mean(torch.abs(self.model.conv1[0].weight.grad)), torch.mean(torch.abs(self.model2.conv1[0].weight.grad))))


                # save model
                if epoch % 20 == 0:
                    print('==> Saving model ...')
                    state = {
                        'net': self.model.state_dict(),
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoints_ducim'):
                        os.mkdir('checkpoints_ducim')
                    torch.save(state, './checkpoints_ducim/ducim.pth')

            print('==> Finished Training ...')

    def infer(self, loss_func, thresh=0.99):
        # self.model.eval()
        if self.n_nets == 1:
            # self.model.eval()
            testloader = torch.utils.data.DataLoader(
                self.ds_test, batch_size=self.batch_size_test, shuffle=False)
            return self.calculate_accuracy(testloader)
        #
        #     for i, data in enumerate(testloader, 0):
        #         # get the inputs
        #         inputs, labels = data
        #         # send them to device
        #         inputs = inputs.to(self.device)
        #         labels = labels.to(self.device)
        #
        #         outputs = self.model(inputs)  # forward pass
        #
        #     pass
        #todo: infer with loaded model from pth file
        if self.n_nets == 2:
            self.model.eval()  # put in evaluation mode
            self.model2.eval()
            # np.random.seed(42)
            torch.backends.cudnn.deterministic = True
            # random.seed(1)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            np.random.seed(42)
            testloader1 = torch.utils.data.DataLoader(
                self.ds_test, batch_size=20, shuffle=False)
            testloader2 = torch.utils.data.DataLoader(
                self.ds_test, batch_size=20, shuffle=True)

            naive_pred = 0
            correct_pred = 0
            total = 0
            conf_naive = np.zeros([2, 2], int)
            conf_nn = np.zeros([2, 2], int)
            c = 0
            for i, data in enumerate(zip(testloader1, testloader2), 0):
                # get the inputs
                inputs1, labels1 = data[0]
                inputs2, labels2 = data[1]
                # send them to device
                inputs1 = inputs1.to(self.device)
                labels1 = labels1.to(self.device)
                inputs2 = inputs2.to(self.device)
                labels2 = labels2.to(self.device)

                # forward + backward + optimize
                outputs1 = self.model(inputs1)  # forward pass
                outputs2 = self.model2(inputs2)
                if loss_func == 'cosine_loss':
                    for j in range(outputs1.shape[0]):
                        temp = torch.tile(outputs1[j, :], (outputs1.shape[0], 1))
                        res_tmp, _ = cosine_loss(temp, outputs2, labels1, labels2, flag=1)
                        # res_tmp, _ = MI_cosine_loss(temp, outputs2, labels1, labels2, flag=1)
                        c += 1*(labels1[j].item() == labels2[res_tmp.argmax().item()].item())
                    # res, y = MI_cosine_loss(outputs1, outputs2, labels1, labels2, flag=1)
                    res, y = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1)
                    res_thresh = res.clone()
                    res_thresh[res_thresh > thresh] = 1
                    res_thresh[res_thresh <= thresh] = 0
                    naive_est = torch.zeros_like(y)
                    naive_pred += torch.sum(naive_est == y)
                    correct_pred += torch.sum(res_thresh == y)
                    total += len(y)
                    with torch.no_grad():
                        for i, l in enumerate(y):
                            conf_naive[l.item(), naive_est[i].item()] += 1
                            conf_nn[l.item(), res_thresh.int()[i].item()] += 1
                elif loss_func == 'constructive':
                    constructive_obj = ContrastiveLoss(mode='cosine')
                    for j in range(outputs1.shape[0]):
                        temp = torch.tile(outputs1[j, :], (outputs1.shape[0], 1))
                        res_tmp, _ = constructive_obj.forward(temp, outputs2, labels1, labels2, flag=1)
                        c += 1*(labels1[j].item() == labels2[res_tmp.argmin().item()].item())
                    res, y = constructive_obj.forward(outputs1, outputs2, labels1, labels2, flag=1)
                    res_thresh = res.clone()
                    res_thresh[res_thresh < thresh] = 1
                    res_thresh[res_thresh >= thresh] = 0
                    naive_est = torch.zeros_like(y)
                    naive_pred += torch.sum(naive_est == y)
                    correct_pred += torch.sum(res_thresh == y)
                    total += len(y)
                    with torch.no_grad():
                        for i, l in enumerate(y):
                            conf_naive[l.item(), naive_est[i].item()] += 1
                            conf_nn[l.item(), res_thresh.int()[i].item()] += 1
                else:
                    cos = nn.CosineSimilarity()
                    m = nn.ReLU()
                    res2 = m(1 - cos(outputs1, outputs2))
                    # res2 = torch.abs(1 - cos(outputs1, outputs2))
                    res2[res2 > thresh] = 1
                    res2[res2 < 1] = 0
                    res_gt = 1*(labels1 == labels2)
                    # res2[res2 > thresh] = 1
                    correct_pred += torch.sum(1*(res2 == res_gt))
                    total += len(res2)
            acc = c/total
            # print(c/total)
            # Run.calc_stat(y, res_thresh, 0)
            # print('test accuracy of naive estimator with thresh = {:.2f} is {:3f} %'.format(thresh, 100*(naive_pred/total)))
            # print('test accuracy of neural network with thresh = {:.2f} is {:3f} %'.format(thresh, 100*(correct_pred/total)))
            # print(conf_naive)
            # print(conf_nn)
            return conf_nn, acc

    def calculate_accuracy(self, trainloader):
        self.model.eval()  # put in evaluation mode
        total_correct = 0
        total_rr = 0
        if trainloader.dataset.hash_id is None:
            confusion_matrix = np.zeros([len(np.unique(trainloader.dataset.y)), len(np.unique(trainloader.dataset.y))], int)
        else:
            confusion_matrix = np.zeros([len(trainloader.dataset.hash_id), len(trainloader.dataset.hash_id)], int)
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                soft_max = nn.Softmax(dim=1)
                predicted = torch.argmax(soft_max(outputs.data), 1)
                # sig = nn.Sigmoid() # change
                # predicted = sig(outputs.data) # change
                # predicted[predicted >= 0.5] = 1 # change
                # predicted[predicted < 0.5] = 0 # change
                total_rr += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                for i, l in enumerate(labels):
                    confusion_matrix[l.item(), predicted[i].int().item()] += 1  # change
                    # confusion_matrix[predicted[i].item(), l.item()] += 1

        model_accuracy = total_correct / total_rr * 100
        # print(model_accuracy)
        return model_accuracy, confusion_matrix


class AE(Model):
 #todo: change this to pytorch
    def __init__(self, nbeats=250):
        super(AE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(nbeats, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# autoencoder = AE()


class CNN(nn.Module):

    def __init__(self, num_labels, nbeats, num_chann=[20, 10], ker_size=[10, 10], stride=[2, 2],
                 dial=[1, 1], pad=[0, 0], num_hidden=60):
        super(CNN, self).__init__()
        if num_labels == 2:
            num_labels = 1  # for BCELoss
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, num_chann[0], kernel_size=ker_size[0], stride=stride[0], dilation=dial[0], padding=pad[0]),
            nn.BatchNorm1d(num_chann[0], track_running_stats=False),  # VERY IMPORTANT APPARENTLY
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )
        self.L1 = np.floor(1+(1/stride[0])*(nbeats + 2*pad[0] - dial[0]*(ker_size[0]-1)-1))
        # self.L1 = np.floor(0.5*(nbeats-10) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_chann[0], num_chann[1], kernel_size=ker_size[1], stride=stride[1], dilation=dial[1], padding=pad[1]),
            nn.BatchNorm1d(num_chann[1], track_running_stats=False),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.L2 = np.floor(1+(1/stride[1])*(self.L1 + 2*pad[1] - dial[1]*(ker_size[1]-1)-1))
        self.L2 = torch.as_tensor(self.L2, dtype=torch.int64)
        # self.L2 = torch.tensor(np.floor(0.5*(self.L1-10) + 1), dtype=torch.int64)
        self.fc1 = nn.Sequential(
            nn.Linear(num_chann[1]*self.L2, num_hidden),
            nn.BatchNorm1d(num_hidden, track_running_stats=False),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
            nn.BatchNorm1d(num_labels, track_running_stats=False),
            # nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )

        self.soft_max = nn.Softmax()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # collapse
        out = out.view(out.size(0), -1)
        # linear layer
        out = self.fc1(out)
        # output layer
        out = self.fc2(out)
        # out = self.soft_max(out) #NO NEED
        out = out.squeeze()
        return out


class TruncatedCNN(nn.Module):

    def __init__(self, CNN_list):
        super(TruncatedCNN, self).__init__()

        self.conv1 = CNN_list[0]
        self.conv2 = CNN_list[1]
        self.fc = CNN_list[2]

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # collapse
        out = out.view(out.size(0), -1)
        # linear layer
        out = self.fc(out)

        return out


class AdverserailCNN(nn.Module):

    def __init__(self, nbeats, num_chann=[20, 10], ker_size=[10, 10], stride=[2, 2],
                 dial=[1, 1], pad=[0, 0], num_hidden=60):
        super(AdverserailCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, num_chann[0], kernel_size=ker_size[0], stride=stride[0], dilation=dial[0], padding=pad[0]),
            nn.BatchNorm1d(num_chann[0]),  # VERY IMPORTANT APPARENTLY
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
        )
        self.L1 = np.floor(1+(1/stride[0])*(nbeats + 2*pad[0] - dial[0]*(ker_size[0]-1)-1))
        # self.L1 = np.floor(0.5*(nbeats-10) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_chann[0], num_chann[1], kernel_size=ker_size[1], stride=stride[1], dilation=dial[1], padding=pad[1]),
            nn.BatchNorm1d(num_chann[1]),
            nn.LeakyReLU(),
            # nn.Dropout(0.25)
        )
        self.L2 = np.floor(1+(1/stride[1])*(self.L1 + 2*pad[1] - dial[1]*(ker_size[1]-1)-1))
        self.L2 = torch.as_tensor(self.L2, dtype=torch.int64)
        # self.L2 = torch.tensor(np.floor(0.5*(self.L1-10) + 1), dtype=torch.int64)
        self.fc1 = nn.Sequential(
            nn.Linear(num_chann[1]*self.L2, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.soft_max = nn.Softmax(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # collapse
        out = out.view(out.size(0), -1)
        # linear layer
        out = self.fc1(out)
        # out = self.soft_max(out)

        return out

class Advrtset(nn.Module):

    def __init__(self, nbeats, num_chann=[20, 10, 30], ker_size=10, stride=2,
                 dial=1, pad=0, drop_out=0.15, num_hidden=[60, 40]):
        super(Advrtset, self).__init__()

        inputs = [num_chann, ker_size, stride, dial, pad, drop_out, num_hidden]
        # outputs = tuple([[x] for x in inputs if type(x) != list])
        for i, x in enumerate(inputs):
            if type(x) != list:
                inputs[i] = [x]
        num_chann, ker_size, stride, dial, pad, drop_out, num_hidden = tuple(inputs)
        inputs = [ker_size, stride, dial, pad]
        if len(num_chann) > 1:
            for i, x in enumerate(inputs):
                if len(x) == 1:
                    inputs[i] = x*len(num_chann)
        ker_size, stride, dial, pad = tuple(inputs)
        w = [len(p) for p in inputs]
        if not(w == [len(num_chann)]*len(w)):
            raise Exception('One of the convolution components does not equal number of channels')
        if len(drop_out) == 1:
            drop_out *= len(num_chann) + len(num_hidden)

        self.num_hidden = num_hidden
        self.conv = nn.ModuleList()
        self.soft_max = nn.Softmax(dim=1)

        self.conv.append(nn.Sequential(
            nn.Conv1d(1, num_chann[0], kernel_size=ker_size[0], stride=stride[0], dilation=dial[0], padding=pad[0]),
            nn.BatchNorm1d(num_chann[0]),  # VERY IMPORTANT APPARENTLY
            nn.LeakyReLU(),
            # nn.Dropout(drop_out[0]),
        ))
        L = np.floor(1+(1/stride[0])*(nbeats + 2*pad[0] - dial[0]*(ker_size[0]-1)-1))

        for idx in range(1, len(num_chann)):
            self.conv.append(nn.Sequential(
                nn.Conv1d(num_chann[idx - 1], num_chann[idx], kernel_size=ker_size[idx], stride=stride[idx], dilation=dial[idx], padding=pad[idx]),
                nn.BatchNorm1d(num_chann[idx]),
                nn.LeakyReLU(),
                # nn.Dropout(drop_out[idx)
            ))
            L = np.floor(1+(1/stride[idx])*(L + 2*pad[idx] - dial[idx]*(ker_size[idx]-1)-1))
        L = torch.as_tensor(L, dtype=torch.int64)
        if len(num_chann) == 1:
            idx = 0
        self.conv.append(nn.Sequential(
            nn.Linear(num_chann[idx]*L, num_hidden[0]),
            nn.BatchNorm1d(num_hidden[0]),
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(idx + 1)
        ))
        #todo: avoid activation function in last layer?
        for idx_lin in range(1, len(num_hidden)):
            self.conv.append(nn.Sequential(
                nn.Linear(num_hidden[idx_lin - 1], num_hidden[idx_lin]),
                nn.BatchNorm1d(num_hidden[idx_lin]),
                nn.LeakyReLU(),
                # nn.ReLU(),
                # nn.Dropout(idx + idx_lin + 1)
            ))

    def forward(self, x):
        out = self.conv[0](x)
        for i in range(1, len(self.conv) - len(self.num_hidden)):  # todo: check if range is true
            out = self.conv[i](out)

        # collapse
        out = out.view(out.size(0), -1)
        for i in range(len(self.conv) - len(self.num_hidden), len(self.conv)):  # todo: check if range is true
        # linear layer
            out = self.conv[i](out)  # NOTICE THAT HERE IT IS NOT CONVOLUTION BUT MLP
        # out = self.soft_max(out)

        return out


class TruncCNNtest(nn.Module):
    def __init__(self, CNN_list):
        super(TruncCNNtest, self).__init__()
        self.conv = CNN_list
    # todo: find the linear layers in CNN_list to decide when to collapse the net

    def forward(self, x):
        out = self.conv[0](x)
    # for i in range(1, len(self.conv) - len(self.num_hidden)):  # todo: check if range is true
    #     out = self.conv[i](out)
    #
    # # collapse
    # out = out.view(out.size(0), -1)
    # for i in range(len(self.conv) - len(self.num_hidden), len(self.conv)):  # todo: check if range is true
    #     # linear layer
    #     out = self.conv[i](out)
    # out = self.soft_max(out)

        return out
