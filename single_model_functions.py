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
import pickle
from comp2ecg_single_model import HRVDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as MinMax
from sklearn import metrics


def load_datasets(full_pickle_path: str, med_mode: str = 'c', mode: int = 0, feat2drop: list = [], sig_type: str = 'rr',
                  train_mode: bool = True) -> object:
    """
    This function is used for loading pickls of training and proper testing prepared ahead and rearrange them as
     HRVDataset objects. It is recommended to have pickle with all jrv features since we can drop whatever we want here.
    :param full_pickle_path: fullpath of training pickle or testing pickle (including '.pkl' ending).
    :param med_mode: type of treatment: 'c' for control and 'a' for abk.
    :param mode: see HRVDataset.
    :param feat2drop: HRV features to drop.
    :param sig_type: rr or HRV signal (The HRV is in koopman mode for now).
    :param train_mode: extract train or test.
    :return: HRVDataset object
    """
    if sig_type == 'rr':
        with open(full_pickle_path, 'rb') as f:
            e = pickle.load(f)
            if train_mode:
                x = e.x_train_specific
                y = e.y_train_specific
                print('Ages used in training set are {}'. format(np.unique(y['age'])))
            else:
                x = e.x_test_specific
                y = e.y_test_specific
                print('Ages used in training set are {}'. format(np.unique(y['age'])))
        x_c, x_a = x[:, y['med'] == 0], x[:, y['med'] == 1]
        y_c, y_a = y[['id', 'age']][y['med'] == 0].values, y[['id', 'age']][y['med'] == 1].values
        if med_mode == 'c':
            dataset = HRVDataset(x_c.T, y_c, mode=mode)  # transpose should fit HRVDataset
        elif med_mode == 'a':
            dataset = HRVDataset(x_a.T, y_a, mode=mode)  # transpose should fit HRVDataset
        else:  # other medications
            raise NotImplementedError
    else:  # Koopman HRV
        #todo: add age tags
        with open(full_pickle_path, 'rb') as f:
            e = pickle.load(f)
            data = e[0:4]
        x_c, x_a = (data[0], data[1])  # (data[2], data[3]) are the y of Koopman, meaning the samples in the future
        y_c, y_a = (x_c.index, x_a.index)
        if med_mode == 'c':
            label_dataset = y_c
            if len(feat2drop) != 0:
                x_c.drop(feat2drop, axis=1, inplace=True)
            np_dataset = np.array(x_c.values, dtype=np.float)
            dataset = HRVDataset(np_dataset, label_dataset, mode=mode)
        elif med_mode == 'a':
            label_dataset = y_a
            if len(feat2drop) != 0:
                x_a.drop(feat2drop, axis=1, inplace=True)
            np_dataset = np.array(x_a.values, dtype=np.float)
            dataset = HRVDataset(np_dataset, label_dataset, mode=mode)
        else:  # other medications
            raise NotImplementedError
    return dataset


def split_dataset(dataset: object, val_size: float = 0.2, seed: int = 42, proper: bool = True) -> tuple:
    """
    This function splits the training dataset into training and validation.
    :param dataset: HRVDataset of training.
    :param val_size: validation size which is a fraction in the range of (0,1).
    :param seed: seed for random choice of mice.
    :param proper: if True the split is done like in real testing, i.e. the training and validation sets contain data
     from different mice. If False, then only the examples are different (came form different time windows).
    :return: 4 numpy arrays.
    """
    if proper:  # meaning splitting train and val into different mice or just different time windows
        np.random.seed(seed)
        tags = np.unique(dataset.y)
        val_tags = np.random.choice(tags, int(np.floor(val_size*len(tags))), replace=False)
        train_tags = np.setdiff1d(tags, val_tags)
        train_mask = np.isin(dataset.y[:, 0], train_tags)
        val_mask = np.isin(dataset.y[:, 0], val_tags)
        x_train = dataset.x[train_mask, :]
        y_train = dataset.y[train_mask, :]
        x_val = dataset.x[val_mask, :]
        y_val = dataset.y[val_mask, :]
    else:  #todo: check if split is done correctly here
        x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, test_size=val_size)
    return x_train, y_train, x_val, y_val


def scale_dataset(*args, input_scaler=None, mode=0, should_scale: bool = False) -> tuple:
    """
    Scaling the data properly according to the training set. If split is made, then scaling is performed for training
    set and then validation and testing are scaled by the same fitted scaler. This means that we call the function twice;
    once for scaling training and validation (len(args)==4) and once for testing (len(args)==1). Second option is to use
    the full dataset for final tuning after choosing our best hyperparmeters and then we call this function once with
    HRVDatasets of training and testing.
    :param args: Can either [output of split_datasets (4 numpy arrays)] or [HRVdataset(train) and HRVdataset(test)] or
    HRVdataset(test).
    :param input_scaler: fitted sklearn MinMax scaler.
    :param mode: see HRVDataset.
    :param should_scale: More for rr. Notice if transpose is needed or not.
    :return: Three options: 1) two scaled HRVDatasets (training & validation) and a fitted scaler.
                            2) two scaled HRVDatasets (training & testing).
                            3) One scaled HRVdataset (testing).
    """
    if input_scaler is None:
        scaler = MinMax()
        if len(args) == 4:  # x_train, y_train, x_val, y_val
            if should_scale:
                x_train = scaler.fit_transform(args[0])
                x_val = scaler.transform(args[2])
            else:  #todo: remove mean rr from every example
                x_train = args[0]
                x_val = args[2]
            return HRVDataset(x_train, args[1], mode=mode), HRVDataset(x_val, args[3], mode=mode), scaler
        elif len(args) == 2:  # HRVdataset(train) and HRVdataset(test)
            if should_scale:
                args[0].x = scaler.fit_transform(args[0].x)  # Notice that this won't do anything if training is already scale
                args[1].x = scaler.transform(args[1].x)
            return args
    else:
        if len(args) != 1:
            raise Exception('Only test set can be an input')
        else:
            if should_scale:
                args[0].x = input_scaler.transform(args[0].x)
            return args


def train_model(model: object, p: object, *args):
    """
    Training the model.
    :param model: chosen neural network.
    :param p: ProSet (project setting) object.
    :param args: Can be either [optimizer, trainloader1, trainloader1, valloader1, valloader2] or
                                [optimizer, trainloader1, trainloader1]
    :param calc_metric: caluclate metrics such as FAR, FPR etc.
    :return: void (model is learned).
    """
    model.train()
    train_epochs(model, p, *args)  # without "*" it would have built a tuple in a tuple


def train_epochs(model: object, p: object, *args):
    """
    This function runs the model in epochs and evaluates the validation sets if exist (see more details in scale_dataset).
    :param: inputs from train_model function.
    :return: prints logs of epochs.
    """
    training_err_vector = np.zeros(p.num_epochs)
    val_err_vector = np.zeros(p.num_epochs)
    for epoch in range(1, p.num_epochs + 1):
        epoch_time = time.time()
        if p.calc_metric:
            training_loss, training_err_vector[epoch - 1] = train_batches(model, p, epoch, *args)
            validation_loss, val_err_vector[epoch - 1] = eval_model(model, p, epoch, *args)
        else:
            training_loss = train_batches(model, p, epoch, *args)
            validation_loss = eval_model(model, p, epoch, *args)
        training_loss /= len(args[1])  # len of trainloader
        if len(args) > 3:  # meaning validation exists.
            validation_loss /= len(args[3])  # len of valloader
        if p.calc_metric:
            log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}  |  Training ERR: {:.4f}  |" \
                  "  Validation ERR: {:.4f}  |  ".format(epoch, training_loss, validation_loss, training_err_vector[epoch - 1],
                                                    val_err_vector[epoch - 1])
        else:
            log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}   ".format(epoch, training_loss, validation_loss)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
    if p.calc_metric:
        idx_val_min = np.argmin(val_err_vector)
        print('Minimal validation ERR was {:.3f} in epoch number {}. Training ERR at the same epoch was: {:.3f}'
              .format(np.min(val_err_vector), 1 + idx_val_min, training_err_vector[idx_val_min]))
        plt.plot(np.arange(1, p.num_epochs + 1), training_err_vector)
        plt.plot(np.arange(1, p.num_epochs + 1), val_err_vector)
        plt.legend(['Train, Test'])
        plt.ylabel('ERR')
        plt.xlabel('epochs')
        plt.show()
    return


def train_batches(model, p, epoch, *args) -> float:
    """
    This function runs over the mini-batches in a single complete epoch using cosine loss.
    :param: inputs from train_model function.
    :param epoch: current epoch to check if pretraining is over
    :return: accumalting loss over the epoch.
    """
    if len(args) == 3:  # validation does not exist
        optimizer, dataloader1, dataloader2 = args
    else:
        optimizer, dataloader1, dataloader2, _, _ = args
    running_loss = 0.0
    scores_list = []
    y_list = []
    for i, data in enumerate(tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)), 0):
        # get the inputs
        inputs1, labels1 = data[0]
        inputs2, labels2 = data[1]
        # send them to device
        inputs1 = inputs1.to(p.device)
        labels1 = labels1.to(p.device)
        inputs2 = inputs2.to(p.device)
        labels2 = labels2.to(p.device)
        # forward
        if epoch > p.pretraining_epoch:
            outputs1, aug_loss1 = model(inputs1, flag_aug=True)  # forward pass
            outputs2, aug_loss2 = model(inputs2, flag_aug=True, y=labels1[:, 1])
            # backward + optimize
            loss = p.reg_aug*(aug_loss1 + aug_loss2) + cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag,
                                                                   lmbda=p.lmbda, b=p.b)

        else:
            outputs1 = model(inputs1)  # forward pass
            outputs2 = model(inputs2)
            # backward + optimize
            loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # backpropagation
        optimizer.step()
        # accumulate mean loss
        running_loss += loss.data.item()
        res_temp, y_temp = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1, lmbda=p.lmbda, b=p.b)
        scores_list.append(0.5 * (res_temp + 1))  # making the cosine similarity as probability
        y_list.append(y_temp)
    if p.calc_metric:
        err = calc_metric(scores_list, y_list, epoch)
        return running_loss, err
    return running_loss


def calc_metric(scores_list, y_list, epoch, train_mode='Training'):
    """
    This function calculates metrics relevant to verification task such as FAR, FRR, ERR, confusion matrix etc.
    :param scores_list: list of tensors (mini-batches) that are probability-like.
    :param y_list: list of tensors (mini-batches) where every example can have the value of 0 (not verified) or 1 (verified).
    :param epoch: current epoch number.
    :param train_mode: Training/Testing for title.
    :return: ERR + plotting every 10 epochs and priniting confusion matrix every epoch.
    """
    scores = torch.cat(scores_list)
    y = torch.cat(y_list)
    fpr, tpr, thresholds = metrics.roc_curve(y.detach().cpu(), scores.detach().cpu())
    # https://stats.stackexchange.com/questions/272962/are-far-and-frr-the-same-as-fpr-and-fnr-respectively
    far, frr, = fpr, 1 - tpr  # since frr = fnr
    # thresholds -= 1
    tr = np.flip(thresholds)
    err_idx = np.argmin(np.abs(frr - far))
    err = 0.5 * (frr[err_idx] + far[err_idx])
    optimal_thresh = tr[err_idx]
    res = scores
    res[scores >= optimal_thresh] = 1
    res[scores < optimal_thresh] = 0
    conf_mat = np.zeros((2, 2))
    conf = (res == y)
    conf_mat[0, 0] += torch.sum(1*(conf[res == 0] == 1))
    conf_mat[0, 1] += torch.sum(1*(conf[res == 1] == 0))
    conf_mat[1, 0] += torch.sum(1*(conf[res == 0] == 0))
    conf_mat[1, 1] += torch.sum(1*(conf[res == 1] == 1))
    print(conf_mat)
    if np.mod(epoch, 10) == 0:
        plt.plot(tr, far, tr, frr)
        plt.legend(['FAR', 'FRR'])
        plt.xlim((tr.min(), 1.2))
        plt.title('{} mode: ERR = {:.2f}, epoch = {}'.format(train_mode, err, epoch))
        plt.show()
    return err


def eval_model(model, p, epoch, *args):
    """
    This function evaluates the current learned model on validation set in every epoch or on testing set in a "single
    epoch".
    :param args: can be either [optimizer, trainloader1, trainloader1, valloader1, valloader2]
                            or [tesloader1, testloader2].
    :param: see train_model.
    :return:
    """
    if len(args) == 3:  # no validation
        eval_loss = 0
        return eval_loss
    else:
        model.eval()
        eval_loss = eval_batches(model, p, epoch, *args)  # without "*" it would have built a tuple in a tuple
        return eval_loss


def eval_batches(model, p,  epoch, *args) -> float:
    """
    This function runs evaluation over batches.
    :param: see eval_model
    :return: accumulating loss over an epoch
    """
    if len(args) == 5:  # validation (thus training is also there)
        _, _, _, dataloader1, dataloader2 = args
    else:  # (=2) only testing
        dataloader1, dataloader2 = args
    running_loss = 0.0
    scores_list = []
    y_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)), 0):
            # get the inputs
            inputs1, labels1 = data[0]
            inputs2, labels2 = data[1]
            # send them to device
            inputs1 = inputs1.to(p.device)
            labels1 = labels1.to(p.device)
            inputs2 = inputs2.to(p.device)
            labels2 = labels2.to(p.device)

            if epoch > p.pretraining_epoch:
                outputs1, aug_loss1 = model(inputs1, flag_aug=True)  # forward pass
                outputs2, aug_loss2 = model(inputs2, flag_aug=True)
                # backward + optimize
                loss = p.reg_aug*(aug_loss1 + aug_loss2) + cosine_loss(outputs1, outputs2, labels1, labels2,
                                                                       flag=p.flag, lmbda=p.lmbda, b=p.b)
            else:
                outputs1 = model(inputs1)  # forward pass
                outputs2 = model(inputs2)
                # backward + optimize
                loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
            # forward
            running_loss += loss.data.item()
            res_temp, y_temp = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1, lmbda=p.lmbda, b=p.b)
            scores_list.append(0.5 * (res_temp + 1))  # making the cosine similarity as probability
            y_list.append(y_temp)
        if p.calc_metric:
            err = calc_metric(scores_list, y_list, epoch, train_mode='Testing')
            return running_loss, err
    return running_loss



