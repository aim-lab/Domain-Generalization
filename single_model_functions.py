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


def load_datasets(full_pickle_path: str, med_mode: str = 'c', mode: int = 0, feat2drop: list = []) -> object:
    """
    This function is used for loading pickls of training and proper testing prepared ahead and rearrange them as
     HRVDataset objects. It is recommended to have pickle with all jrv features since we can drop whatever we want here.
    :param full_pickle_path: fullpath of training pickle or testing pickle (including '.pkl' ending).
    :param med_mode: type of treatment: 'c' for control and 'a' for abk.
    :param mode: see HRVDataset.
    :param feat2drop: HRV features to drop.
    :return: HRVDataset object
    """
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
    :return: 4 numpy arrays (2 matrices and 2 vectors).
    """
    if proper:  # meaning splitting train and val into different mice or just different time windows
        np.random.seed(seed)
        tags = np.unique(dataset.y)
        val_tags = np.random.choice(tags, int(np.floor(val_size*len(tags))), replace=False)
        train_tags = np.setdiff1d(tags, val_tags)
        train_mask = np.isin(dataset.y, train_tags)
        val_mask = np.isin(dataset.y, val_tags)
        x_train = dataset.x[train_mask, :]
        y_train = dataset.y[train_mask]
        x_val = dataset.x[val_mask, :]
        y_val = dataset.y[val_mask]
    else:
        x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, test_size=val_size)
    return x_train, y_train, x_val, y_val


def scale_dataset(*args, input_scaler=None, mode=0) -> tuple:
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
    :return: Three options: 1) two scaled HRVDatasets (training & validation) and a fitted scaler.
                            2) two scaled HRVDatasets (training & testing).
                            3) One scaled HRVdataset (testing).
    """
    if input_scaler is None:
        scaler = MinMax()
        if len(args) == 4:  # x_train, y_train, x_val, y_val
            x_train = scaler.fit_transform(args[0])
            x_val = scaler.transform(args[2])
            return HRVDataset(x_train, args[1], mode=mode), HRVDataset(x_val, args[3], mode=mode), scaler
        elif len(args) == 2:  # HRVdataset(train) and HRVdataset(test)
            args[0].x = scaler.fit_transform(args[0].x)  # Notice that this won't do anything if training is already scale
            args[1].x = scaler.transform(args[1].x)
            return args
    else:
        if len(args) != 1:
            raise Exception('Only test set can be an input')
        else:
            args[0].x = input_scaler.transform(args[0].x)
            return args


def train_model(model: object, p: object, *args, calc_metric: bool = False):
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
    train_epochs(model, p, calc_metric, *args)  # without "*" it would have built a tuple in a tuple


def train_epochs(model: object, p: object, calc_metric: bool, *args):
    """
    This function runs the model in epochs and evaluates the validation sets if exist (see more details in scale_dataset).
    :param: inputs from train_model function.
    :return: prints logs of epochs.
    """
    for epoch in range(1, p.num_epochs + 1):
        epoch_time = time.time()
        training_loss = train_batches(model, p, calc_metric, *args)
        training_loss /= len(args[1])  # len of trainloader
        validation_loss = eval_model(model, p, calc_metric=calc_metric, *args)
        if len(args) > 3:  # meaning validation exists.
            validation_loss /= len(args[3])  # len of valloader
        log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}   ".format(epoch, training_loss, validation_loss)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)


def train_batches(model, p, calc_metric, *args) -> float:
    """
    This function runs over the mini-batches in a single complete epoch using cosine loss.
    :param: inputs from train_model function.
    :return: accumalting loss over the epoch.
    """
    if len(args) == 3:  # validation does not exist
        optimizer, dataloader1, dataloader2 = args
    else:
        optimizer, dataloader1, dataloader2, _, _ = args
    running_loss = 0.0
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
        outputs1 = model(inputs1)  # forward pass
        outputs2, aug_loss = model(inputs2, flag_aug=True)
        # backward + optimize
        loss = aug_loss + cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # backpropagation
        optimizer.step()
        # accumulate mean loss
        running_loss += loss.data.item()
        if calc_metric:
            #todo: use built-in sklean metrics report
            raise NotImplementedError
    return running_loss


def eval_model(model, p, *args, calc_metric=0):
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
        eval_loss = eval_batches(model, p, calc_metric, *args)  # without "*" it would have built a tuple in a tuple
        return eval_loss


def eval_batches(model, p, calc_metric, *args) -> float:
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
            # forward
            outputs1 = model(inputs1)  # forward pass
            outputs2, aug_loss = model(inputs2, flag_aug=True)
            # aug_loss = torch.tensor(aug_loss).unsqueeze(0)
            # calculate loss
            loss = aug_loss + cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
            running_loss += loss.data.item()
            if calc_metric:
                raise NotImplementedError
    return running_loss


