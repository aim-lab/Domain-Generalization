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
from single_model_functions import *
from project_settings import ProSet

if __name__ == '__main__':
    p = ProSet()

    # wandb.login()
    # wandb.init('test', entity=p.entity)
    # config = dict(n_epochs=p.num_epochs, batch_size=p.batch_size)

    tr_dataset_1 = load_datasets(p.train_path, feat2drop=['Age'])
    x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_1, proper=p.proper)
    tr_dataset_1, val_dataset_1, scaler1 = scale_dataset(x_tr, y_tr, x_val, y_val)
    tr_dataset_2 = load_datasets(p.train_path, feat2drop=['Age'], mode=1)
    x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_2, proper=p.proper)
    tr_dataset_2, val_dataset_2, scaler2 = scale_dataset(x_tr, y_tr, x_val, y_val, mode=1)

    ##### NOTICE  WHEN TO USE SCALER FOR TESTING. IF WE USE FULL DATASET FOR LEARNING THUS OUTSIDE SCALER IS NOT NEEDED.

    model = Advrtset(tr_dataset_1.x.shape[1], p, ker_size=p.ker_size, stride=p.stride, dial=p.dial).to(p.device)
    if p.mult_gpu:
        model = nn.DataParallel(model, device_ids=p.device_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    ############## TRAINING SET ##########################
    trainloader1 = torch.utils.data.DataLoader(
        tr_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0)
    trainloader2 = torch.utils.data.DataLoader(
        tr_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0)
    ############## VALIDATION SET ###########################
    valloader1 = torch.utils.data.DataLoader(
        val_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0)
    valloader2 = torch.utils.data.DataLoader(
        val_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0)

    train_model(model, p, optimizer, trainloader1, trainloader2, valloader1, valloader2)

    # wandb.finish()
