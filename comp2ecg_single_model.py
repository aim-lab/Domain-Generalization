import os
import numpy as np
import torch
import torch.nn as nn
from data_loader import TorchDataset
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
from deep_models import Advrtset
from deep_models import cosine_loss
from tqdm import tqdm
from torch.utils.data import Dataset
import wandb




class HRVDataset(Dataset):

    def __init__(self, x, y, mode=0):
        # self.x = x.clone().detach().requires_grad_(True)
        super().__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.mode = mode


    def __len__(self):
        return self.x.shape[0]  # since we transpose

    def __getitem__(self, idx):
        """
        Two datasets are build for Siamease network. We have to make sure that within a batch
        the comparison is made 50% of the time with negative example and 50% with negative. The two
        datasets have shuffle=False. mode=1 will be only for the second dataset
        :param idx: index of sample
        :param mode: used for second dataset to have 50% of the mice compared to the first dataset to be different
        :return:
        """
        # np.random.seed(5)
        if self.mode:
            y = self.y[idx, 0]
            r = np.random.randint(2)  # 50%-50%
            if r:  # find negative example
                neg_list = np.argwhere(self.y[:, 0] != y)
                idx = neg_list[np.random.randint(0, len(neg_list))].item()
            else:
                idx_temp = None
                pos_list = np.argwhere(self.y[:, 0] == y)
                while (idx_temp is None) or (idx_temp == idx):  # avoid comparing the same signals
                    idx_temp = pos_list[np.random.randint(0, len(pos_list))].item()
                idx = idx_temp
        x = self.x[idx:idx+1, :]
        y = self.y[idx, :]
        sample = (torch.from_numpy(x).requires_grad_(True).type(torch.FloatTensor), torch.from_numpy(y).type(torch.IntTensor))  # just here convert to torch
        return sample


if __name__ == '__main__':
    with open('/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/x_y.pkl', 'rb') as f:
        e = pickle.load(f)
        data_tr = e[0:4]
        max_age = e[-1]
    x_tr_c = data_tr[0]
    x_tr_a = data_tr[1]
    y_tr_c = x_tr_c.index
    y_tr_a = x_tr_a.index
    with open('/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/no_exp_test.pkl', 'rb') as f:
        e = pickle.load(f)
        data_ts = e[0:4]
        max_age_test = e[-1]
    x_ts_c = data_ts[0]
    x_ts_a = data_ts[1]
    y_ts_c = x_ts_c.index
    y_ts_a = x_ts_a.index

    batch_size = 2 ** 10
    b = -0.8
    lmbda = 1000
    flag = 0
    # dataset = torch.from_numpy(data_tr[0].to_numpy().T)  # now every column is a feature vector
    # dataset.type(torch.float32)
    # dataset_X1 = torch.from_numpy(np.array(data_tr[0].values, dtype=np.float))
    # dataset_X2 = torch.from_numpy(np.array(data_tr[0].values, dtype=np.float))
    dataset_X1 = np.array(x_tr_c.values, dtype=np.float)
    # dataset_X2 = np.array(x_tr_c.values, dtype=np.float)
    dataset_label = y_tr_c

    dataset_1 = HRVDataset(dataset_X1, dataset_label)
    dataset_2 = HRVDataset(dataset_X1, dataset_label, mode=1)
    # dataset_1 = HRVDataset(dataset_X1.type(torch.FloatTensor), dataset_label)
    # dataset_2 = HRVDataset(dataset_X2.type(torch.FloatTensor), dataset_label, mode=1)
    # dataset = torch.tensor(np.array(dataset, dtype=float))

    num_epochs = 4
    lr = 0.001
    phi = np.pi
    # Device configuration, as before

    # torch.backends.cuda.matmul.allow_tf32
    # create model, send it to device
    device_ids = [1, 3, 4, 5, 6]
    device = torch.device('cuda:' + str(device_ids[0]) if torch.cuda.is_available() else 'cpu')
    model2 = Advrtset(dataset_X1.shape[1], ker_size=2, stride=1, dial=1).to(device)
    parallel_net_2 = nn.DataParallel(model2, device_ids=device_ids)
    optimizer_2 = torch.optim.Adam(parallel_net_2.parameters(), lr=lr, weight_decay=1) #, momentum=0.9)

     # put in training mode
    parallel_net_2.train()


    trainloader1 = torch.utils.data.DataLoader(
        dataset_1, batch_size=batch_size, shuffle=False, num_workers=0)
    trainloader2 = torch.utils.data.DataLoader(
        dataset_2, batch_size=batch_size, shuffle=False, num_workers=0)
    for epoch in range(1, num_epochs + 1):
        epoch_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(tqdm(zip(trainloader1, trainloader2), total=len(trainloader1)), 0):
            # get the inputs

            inputs1, labels1 = data[0]
            inputs2, labels2 = data[1]
            # send them to device
            inputs1 = inputs1.to(device)
            labels1 = labels1.to(device)
            inputs2 = inputs2.to(device)
            labels2 = labels2.to(device)

            # forward + backward + optimize
            outputs1 = parallel_net_2(inputs1)  # forward pass
            outputs2 = parallel_net_2(inputs2)
            loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=flag, lmbda=lmbda, b=b)
            # zero the parameter gradients
            optimizer_2.zero_grad()
            loss.backward(retain_graph=True)  # backpropagation
            # update parameters
            optimizer_2.step()
            running_loss += loss.data.item()


        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader1)
        log = "Epoch: {} | Loss: {:.4f}  | ".format(epoch, running_loss)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)

    # infer
    # put in training mode
    parallel_net_2.eval()
    testset_X1 = np.array(x_ts_c.values, dtype=np.float)
    testset_X2 = np.array(x_ts_c.values, dtype=np.float)
    testset_label = y_ts_c
    tetset_1 = HRVDataset(testset_X1, testset_label)
    tetset_2 = HRVDataset(testset_X2, testset_label, mode=1)
    testloader1 = torch.utils.data.DataLoader(
        tetset_1, batch_size=batch_size, shuffle=False)
    testloader2 = torch.utils.data.DataLoader(
        tetset_2, batch_size=batch_size, shuffle=False)

    thresh = np.linspace(-1, 1, 30)
    conf_mat_list = []
    acc = np.zeros(len(thresh))
    far = np.zeros(len(thresh))
    frr = np.zeros(len(thresh))
    for idx, tr in enumerate(thresh):
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
            inputs1 = inputs1.to(device)
            labels1 = labels1.to(device)
            inputs2 = inputs2.to(device)
            labels2 = labels2.to(device)

            # forward + backward + optimize
            outputs1 = parallel_net_2(inputs1)  # forward pass
            outputs2 = parallel_net_2(inputs2)
            # if loss_func == 'cosine_loss':
            for j in range(outputs1.shape[0]):
                temp = torch.tile(outputs1[j, :], (outputs1.shape[0], 1))
                res_tmp, _ = cosine_loss(temp, outputs2, labels1, labels2, flag=1, lmbda=lmbda, b=b)
                # res_tmp, _ = MI_cosine_loss(temp, outputs2, labels1, labels2, flag=1)
                c += 1*(labels1[j].item() == labels2[res_tmp.argmax().item()].item())
            # res, y = MI_cosine_loss(outputs1, outputs2, labels1, labels2, flag=1)
            res, y = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1, lmbda=lmbda, b=b)
            res_thresh = res.clone()
            res_thresh[res_thresh > tr] = 1
            res_thresh[res_thresh <= tr] = 0
            naive_est = torch.zeros_like(y)
            naive_pred += torch.sum(naive_est == y)
            correct_pred += torch.sum(res_thresh == y)
            total += len(y)
            with torch.no_grad():
                for i, l in enumerate(y):
                    conf_naive[l.item(), naive_est[i].item()] += 1  # i.e. rows are gt and columns are preds
                    conf_nn[l.item(), res_thresh.int()[i].item()] += 1  # i.e. rows are gt and columns are preds
        conf_mat_list.append(conf_nn)
        s = conf_nn.sum(axis=1)
        far[idx] = conf_nn[0, 1]/s[0]
        frr[idx] = conf_nn[1, 0]/s[1]
        acc[idx] = c/total
    plt.plot(thresh, far)
    plt.plot(thresh, frr)
    plt.legend(['FAR', 'FRR'])
    plt.show()
    a=1




    # performs a forward pass
    # predictions = parallel_net(inputs)
    # # computes a loss function
    # loss = loss_function(predictions, labels)
    # # averages GPU-losses and performs a backward pass
    # loss.mean().backward()
    # optimizer.step()
    a=1


