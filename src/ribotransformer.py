###############################################################################
########
###############################################################################
from pathlib import Path
import pandas as pd
import argparse
import torch as pt
import torch.nn as nn
import numpy as np
import ipdb
import time
import torch
import math
import scipy.stats

ten = torch.Tensor


def norm_ribosig(data2norm):
    data2norm.ribosignal.shape

    data2norm.ribosignal -= (data2norm.ribosignal.mean(axis=1).
                             reshape([-1, 1]))
    data2norm.ribosignal /= (data2norm.ribosignal.std(axis=1).
                             reshape([-1, 1]))
    return data2norm


def clip_norm_ribosig(data2norm, lclip=0, rclip=0):
    clipsig = data2norm.ribosignal.numpy().copy()
    cliptensor = data2norm.orfclip(data2norm.data['codons'], lclip, rclip)
    gnum = clipsig.shape[0]
    orfmeans = np.array([clipsig[i][cliptensor[i]].mean()
                         for i in range(gnum)])
    clipsig /= orfmeans.reshape([-1, 1])
    clipsig[~cliptensor] = NULLVAL
    data2norm.ribosignal = pt.Tensor(clipsig)
    return data2norm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model_orig, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = 2 * int(d_model_orig / 2)
        pe = pt.zeros(max_len, d_model)
        # 5000,200
        position = pt.arange(0, max_len, dtype=pt.float).unsqueeze(1)
        # 5000,1
        div_term = pt.exp(pt.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # shape 100
        pe[:, 0::2] = pt.sin(position * div_term)
        pe[:, 1::2] = pt.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        if not pe.shape[2] == d_model_orig:
            pe = pt.cat([pe, pe[:, :, 0:1]], axis=2)
            pe[:, :, -1] = 0
        print(pe.shape)
        # pe.shape==5000,1,200
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
    Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
    """
        x = x + self.pe[:x.size(0), :, :x.size(2)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.pos_encoder = PositionalEncoding(ninp, dropout=0)
        encoder_layers = TransformerEncoderLayer(
            ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    # src,self=tdata,cmodel

    def forward(self, src):
        src, offset = src
        ixind = src.shape[1]
        ixnos = src[:, (ixind - 1):ixind, :]
        src = src[:, 0:(ixind - 1), :]
        isnull = src[:, 0, :] == 1
        src = src.permute([2, 0, 1])
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # # assert list(output.shape) == (512,n_batch,n_feats)
        output = self.decoder(output)
        # assert list(output.shape) == (512,n_batch,1)
        output = output.squeeze(2).permute([1, 0])
        output[isnull] = NULLVAL
        output = output + ixnos.squeeze(1)
        # output = ixnos.squeeze(1)
        return output


mseloss = nn.MSELoss(reduction='none')
poisloss = nn.PoissonNLLLoss(log_input=True)
poislossnr = nn.PoissonNLLLoss(log_input=True, reduction='none')
#


def test_ribotransmodel():
    # https://pypt.org/tutorials/beginner/transformer_tutorial.html
    # this creates a function that gets a vector of len,1,d_model and adds
    # it to a a tensor of e.g. len,n,d_model
    tdata, ttarget = next(iter(train_data))
    toutput = rtmodel(tdata)

    def get_rdata_cor(target, output, data, train_data):
        datacor = round(scipy.stats.pearsonr(
            target[train_data.orfclip(data[0], lclip, rclip)],
            output[train_data.orfclip(data[0], lclip, rclip)].detach()
        )[0], 3)
        return datacor

    # what's the correlation on this training set?
    # Well it's one if I have ixnos spit back out the training data...
    get_rdata_cor(ttarget, toutput, tdata, train_data)

    assert list(toutput.shape) == [train_data.batch_size, 512]
    assert list((toutput - ttarget).shape) == [train_data.batch_size, 512]

    cliptens = train_data.orfclip(tdata[0])

    # #x,y=output,target
    cliptens = train_data.orfclip(tdata[0])
    mseregcriterion(toutput, ttarget, cliptens)


def mseregcriterion(x, y, cliptens):
    return mseloss(x[cliptens], y[cliptens]).mean()


regcriterion = mseregcriterion


def get_rdata_cor(target, output, data, train_data):
    datacor = round(scipy.stats.pearsonr(
        target[train_data.orfclip(data[0], lclip, rclip)],
        output[train_data.orfclip(data[0], lclip, rclip)].detach()
    )[0], 3)
    return datacor


def get_val_cor(val_data, rtmodel):
    val_data.batchshuffle = False
    rtmodel.eval()
    with pt.no_grad():
        preds = [rtmodel(b[0]).detach()[val_data.orfclip(
            b[0][0], lclip, rclip)] for b in val_data]
        targets = [b[1][val_data.orfclip(b[0][0], lclip, rclip)]
                   for b in val_data]
    datacor = round(scipy.stats.pearsonr(
        pt.cat(preds),
        pt.cat(targets)
    )[0], 4)
    rtmodel.train()
    return datacor


def train(epoch):
    datacors = []
    valcors = []
    rtmodel.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    bptt = train_data.batch_size
    for batch, batchdata in enumerate(train_data):
        # for batch in range(0,1):
        data, target = batchdata
        optimizer.zero_grad()
        output = rtmodel(data)
        cliptens = train_data.orfclip(data[0])
        loss = regcriterion(output, target, cliptens)
        if pt.isnan(loss):
            ipdb.set_trace()
        loss.backward()
        total_loss += loss.item()
        log_interval = 10
        if batch % log_interval == 0 and batch > 0:
            datacor = get_rdata_cor(target, output, data, train_data)
            print(f'batch: {batch}/{len(train_data) // bptt}')
            print(f'Correlation with training set: {round(datacor,4)}')
            vcor = get_val_cor(val_data, rtmodel)
            print(f'Correlation with validation set: {round(vcor,4)}')
            datacors.append(datacor)
            valcors.append(vcor)
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(
                          train_data) // bptt, scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, cur_loss))

            total_loss = 0
            start_time = time.time()
        optimizer.step()
    return datacors, valcors


def save_losses(train_losses, val_losses):
    lossitems = zip([v for sub in train_losses for v in sub],
                    [v for sub in val_losses for v in sub])
    cor_loss_df = pd.DataFrame(list(lossitems))
    epochbatches = (cor_loss_df.shape[0] // 3)
    epoch = [[e] * epochbatches for e in range(3)]
    epoch = [e for sub in epoch for e in sub]
    cor_loss_df['epoch'] = epoch
    cor_loss_df.columns = ['training', 'validation', 'epoch']
    cor_loss_df.to_csv('ix_tr_norm_trans_losses.tsv', sep='\t', index=False)


def train_model():
    val_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch}/{epochs}')
        tr, val = train(epoch)
        val_losses.append(val)
        train_losses.append(tr)
        scheduler.step()
    save_losses(train_losses, val_losses)


def get_elongs(rdata, rtmodel):
    rdata.usedata = datatouse
    rdata.batchshuffle = False
    next(iter(rdata))[0][0].shape
    rtmodel.eval()
    with pt.no_grad():
        preds = [rtmodel(b[0]).detach() for
                 b in rdata]
    preds = pt.cat(preds, axis=0)
    orfclipten = [val_data.orfclip(b[0][0], 0, 0) for b in rdata]
    orfclipten = pt.cat(orfclipten, axis=0)
    orfmeans = [preds[i][orfclipten[i]].mean().item() for
                i in range(preds.shape[0])]
    rtmodel.train()
    nms = list(rdata.gene2num.index)
    nms[1]
    rdata.batchshuffle = True
    return preds, orfmeans, nms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        help="The input Ribotransdata object",
        default='data/ribotranstest_rdata.pt'
    )
    parser.add_argument(
        "-o",
        help='the stem of the output file the program creates',
        default='ribotranstest'
    )
    args = parser.parse_args()
    data_file = Path(args.d)
    outputfile = Path(args.o + '.ribotrans.pt')

    # load data object
    rdata = torch.load(data_file)
    assert 'ixnos' in rdata.data.keys()

    # training params
    NULLVAL = -4
    epochs = 3
    datatouse = ['codons', 'ixnos']
    # clipping numbers for comparison to ixnos
    lclip = 20
    rclip = -20

    # define training validation split
    train_data, test_data, val_data = rdata.split_data(500, 1)
    train_data.usedata = datatouse
    val_data.usedata = datatouse
    test_data.usedata = datatouse

    #  device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    device = "cpu"

    # normalize our data, including NULL values for comparison to ixnos
    print('clipped normalization')
    train_data, val_data, test_data = map(
        clip_norm_ribosig, (train_data, val_data, test_data))

    # define our transformer model
    datadim = train_data.datadim()
    rtmodel = TransformerModel(
        datadim - 1, nhead=5, nhid=10, nlayers=4).to(device)

    # verify the model can evaluate (dimensions etc corret)
    test_ribotransmodel()

    # set our training hyperparameters
    lr = 0.001
    print('learning rate: {:3f}'.format(lr))
    optimizer = pt.optim.Adam(rtmodel.parameters(), lr=lr)
    scheduler = pt.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
    train_data.batch_size = 32
    val_data.batch_size = 32
    print('batch size: {:3f}'.format(train_data.batch_size))

    # train our model
    train_model()

    # Now use our trained model to get predictions for all transcriopts
    rdata.usedata = train_data.usedata
    predslist = get_elongs(rdata, rtmodel)
    elongdf = pd.DataFrame(
        [pd.Series(predslist[1]), pd.Series(predslist[2])]).transpose()
    elongdf.columns = ['elong', 'tr_id']
    elongdf.to_csv(args.o + '_ribotrans_ix_elongs.csv')
