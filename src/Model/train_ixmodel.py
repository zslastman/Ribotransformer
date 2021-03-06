import torch
import ipdb  # ignore
import sys
import argparse
import copy
import ixnosdata
from pathlib import Path
from typing import List
from scipy import stats
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import collections

ipdb

class Net(nn.Module):
    def __init__(self, n_feats, n_hid, finalmodelweights=None):
        super(Net, self).__init__()
        self.h1 = nn.Linear(n_feats, n_hid)  # 5*5 from image dimension
        self.out = nn.Linear(n_hid, 1)
        if finalmodelweights is not None:
            self.init_weights(n_hid, finalmodelweights)

    def init_weights(self, n_hid, finalmodelweights):
        self.h1.weight.data = torch.tensor(finalmodelweights[0].T)
        self.h1.bias.data = torch.tensor(finalmodelweights[1].T)
        self.out.weight.data = (torch.tensor(finalmodelweights[2].T).
                                reshape([1, n_hid]))
        self.out.bias.data = (torch.tensor(finalmodelweights[3].T).
                              reshape([1]))

    # self,x = mlp, X_tr
    def forward(self, x):
        hidden = torch.tanh(self.h1(x))
        out = F.relu(self.out(hidden))
        return out


# ##############################################################################
# #######Let's try it with my own generated data (hopefully teh same)
# ##############################################################################

def lrlambda(epoch, lr_decay=16):
    return 1 / (1 + float(epoch) / lr_decay)


class Dset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

    #
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    #
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.X[index]
        y = self.y[index]
        return X, y


@dataclass
class Trainres:
    beststate: collections.OrderedDict = field(
        default_factory=collections.OrderedDict)
    trainloss: List[float] = field(default_factory=list)
    testloss: List[float] = field(default_factory=list)
    bestloss: float = 1e12
    bestcor: float = -1
    testcors: List[float] = field(default_factory=list)

    def saveastuple(self):
        obtuple = (trainres.beststate,
        trainres.trainloss,
        trainres.testloss,
        trainres.bestloss,
        trainres.bestcor,
        trainres.testcors)

def train(trainloader, net, criterion, optimizer,
          scheduler, n_feats, X_te, y_te, epochs=55):
    trainres = Trainres()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # if i % 20  == 20-1:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                    # (epoch + 1, i + 1, running_loss / 20))
                # running_loss = 0.0
        trainres.trainloss.append(float(running_loss / (1 + i)))
        scheduler.step()
        net.eval()
        with torch.no_grad():
            loss = criterion(net(X_te[:, :n_feats]),
                             y_te.reshape(-1, 1))
        trainres.testloss.append(float(loss))
        myout = net(X_te)
        cor = stats.pearsonr(myout.detach().numpy().flatten(),
                             y_te.flatten())[0]
        sys.stdout.write(
            f'Epoch: {epoch} correlation with testset:\tcor: {cor:.4f}\r')
        trainres.testcors.append(cor)
        if loss < trainres.bestloss:
            trainres.bestloss = loss
            trainres.bestcor = cor
            trainres.beststate = net.state_dict()

    return(trainres)


def train_on_ixdata(ixdataset, epochs=55):

    # first set up the data with trainloader and te datasets
    trainset = Dset(ixdataset.X_tr.float(),
                    ixdataset.y_tr.reshape(-1, 1).float())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                              shuffle=True, num_workers=0)
    i, data = next(enumerate(trainloader, 0))
    # Now set up the neural network, like ixnos
    n_feats = data[0].shape[1]
    n_hid = 200
    net = Net(n_feats, n_hid)
    assert net(data[0]) is not None
    # loss function
    criterion = nn.MSELoss()
    # optimize like ixnos
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001,
                                momentum=0.9, nesterov=True)
    # set learning rate decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrlambda)
    #
    trainres = train(
        trainloader, net, criterion, optimizer, scheduler,
        n_feats, ixdataset.X_te[:, :].float(),
        ixdataset.y_te.float(), epochs=epochs)
    #
    net.load_state_dict(trainres.beststate)
    trainres.model = net
    return trainres


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="Input dataframe with read counts per codon",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default=f'ribotransdata/P0_ribo_2/P0_ribo_2.all.psites.csv'
    )
    parser.add_argument(
        "-c",
        help="Input dataframe with dimensions of csv",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        # default=f'cortexomicstest.cdsdims.csv'
        default=f'ribotransdata/P0_ribo_2/P0_ribo_2.cdsdims.csv'
    )
    parser.add_argument(
        "-o",
        help="path stem for the output saved model",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default=f'ixnosmodels/P0_ribo_2/P0_ribo_2'
    )
    args = parser.parse_args()

    print(args)

    df_file: Path = Path(args.i)
    cdsdims_file: Path = Path(args.c)
    # now read in
    df: pd.DataFrame = pd.read_csv(df_file)
    cdsdims: pd.DataFrame = pd.read_csv(cdsdims_file)
    # df = df.loc[]
    ixdataset = ixnosdata.Ixnosdata(df, cdsdims)
    trainres = train_on_ixdata(ixdataset, epochs=3)
    #
    torch.save(trainres.model, f'{args.o}.bestmodel.pt')
    torch.load(f'{args.o}.bestmodel.pt')