################################################################################
########Very simple convolutional model, this works with fake data really well
########Also learns something with the real yeast data
################################################################################
print_batch_loss = False
NULLVAL=-4
epochs = 3
lclip = 20
rclip = -20
# datatouse = ['codons','seqfeats']
datatouse = ['codons','ixnos']

if not 'rdata' in globals().keys():
    import os
    exec(open('src/Parse/1_ribodataob.py').read())


train_data,test_data,val_data = split_data(rdata,500,1)

# print('shuffling codon data')
# for g in range( train_data.data['codons'].shape[0]):
#     for pos in range( train_data.data['codons'].shape[2]):
#         train_data.data['codons'][g,:,pos]=0
#         train_data.data['codons'][g,np.random.choice(range(64)),pos]=1


if True: 
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import plotext as plx
    import numpy as np
    import random
    import ipdb
    import time
    import scipy.stats
    from torchsummary import summary
    ten = torch.Tensor

    # based on original code from
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    print('defining...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def norm_ribosig(data2norm):
        data2norm.ribosignal.shape

        data2norm.ribosignal -= data2norm.ribosignal.mean(axis=1).reshape([-1,1])
        data2norm.ribosignal /= data2norm.ribosignal.std(axis=1).reshape([-1,1])

        # cliptensor = data2norm.orfclip(data2norm.data['codons'],0,0)
        # data2norm.ribosignal[~cliptensor]=NULLVAL
        # data2norm.offset = None
        return data2norm

    # data2norm=test_data
    def clip_norm_ribosig(data2norm):
        clipsig = data2norm.ribosignal.numpy().copy()
        cliptensor = data2norm.orfclip(data2norm.data['codons'],lclip,rclip)
        # clipsig[~cliptensor]=0
        gnum = clipsig.shape[0]
        orfmeans = np.array([clipsig[i][cliptensor[i]].mean() for i in range(gnum)])
        orfstds = np.array([clipsig[i][cliptensor[i]].std() for i in range(gnum)])
        clipsig -= orfmeans.reshape([-1,1])
        # clipsig /= orfstds.reshape([-1,1])
        clipsig[~cliptensor]= NULLVAL
        data2norm.ribosignal = torch.Tensor(clipsig)
        # data2norm.offset = None
        return data2norm

    # train_data,val_data,test_data = map(norm_ribosig,(train_data,val_data,test_data))
    print('clipped normalization')
    train_data,val_data,test_data = map(clip_norm_ribosig,(train_data,val_data,test_data))
    
    
    # train_data.data['codons'][:,0,:]
    # rsum=train_data.ribosignal[0].sum()
    # train_data.ribosignal

if True:
    #https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    #this creates a function that gets a vector of len,1,d_model and adds it to a a tensor of e.g. len,n,d_model
    # d_model_orig=65
    # self=posenc
    # dropout=0
    # max_len=5000
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model_orig, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            d_model = 2*int(d_model_orig/2)
            pe = torch.zeros(max_len, d_model)
            #5000,200
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            #5000,1
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            #shape 100
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            if not pe.shape[2]==d_model_orig:
                pe = torch.cat([pe,pe[:,:,0:1]],axis=2)
                pe[:,:,-1]=0
            print(pe.shape)
            #pe.shape==5000,1,200
            self.register_buffer('pe', pe)

        def forward(self, x):
            """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)

    # posenc=PositionalEncoding(65,dropout=0)
    # #checking the positional encodings indeed encode position
    # tdata,ttarget=next(iter(train_data))
    # x=tdata[0]
    # x[:,:,:]=0
    # x.shape
    # x=x.permute([2,0,1])
    # x.shape
    # xenc=posenc(x)
    # assert (xenc[2,0,:]==xenc[2,3,:]).all()

    # tdata[0].max()
    # tdatacop=tdata[0].numpy().copy()
    # tdatacop2=tdatacop.copy()
    # tdatacop2 = tdatacop2 - tdatacop2
    # cmodel.pos_encoder(tdata[0])
    # diff = tdata[0] - tdatacop
    # diff[0,:,0]
    # diff[1,:,0]

    # for datatouse in (['codons'],['codons','esm'],['esm']):
    # for datatouse in (['codons']):

    train_data.usedata=datatouse
    val_data.usedata=datatouse
    test_data.usedata=datatouse

    # batch[0][0].shape
    # batch[0][0].permute([2,0,1])
    #ninp is the embedding dimension
    #ninp,nhead,nhid,dropout = 65,10,10,0.5

    class TransformerModel(nn.Module):
        def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
            super(TransformerModel, self).__init__()
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            self.pos_encoder = PositionalEncoding(ninp, dropout=0)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.decoder = nn.Linear(ninp, 1)
            self.init_weights()
        def init_weights(self):
            initrange = 0.1
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)
        # src,self=tdata,cmodel
        def forward(self, src):
            src,offset = src
            ixind = src.shape[1]
            ixnos = src[:,(ixind-1):ixind,:]
            src = src[:,0:(ixind-1),:]
            isnull=src[:,0,:]==1
            # output = self.decoder(src.permute([2,0,1]))
            # src[:,1:,:]=0
            # assert list(src.shape) == [300,65,512] 
            src = src.permute([2,0,1])
            src = self.pos_encoder(src)
            # # output = self.conv(src)
            # # output = output + offset.log()
            # # output[src[:,0:1,:]==1]=0
            # # output = output * offset
            output = self.transformer_encoder(src) 
            # # assert list(output.shape) == (512,n_batch,n_feats)
            output = self.decoder(output)
            # assert list(output.shape) == (512,n_batch,1) 
            output = output.squeeze(2).permute([1,0])
            output[isnull] = NULLVAL
            # output = output + ixnos.squeeze(1)
            return output
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadim = train_data.datadim()
    cmodel = TransformerModel(datadim-1, nhead=5,nhid=10,nlayers=4).to(device)
    tdata,ttarget=next(iter(train_data))
    toutput = cmodel(tdata)

    #test the model
    # assert train_data.usedata == ['codons','esm']j
    # assert train_data.datadim() == 65+1280
    # assert val_data.datadim() == 65+1280
    tdata,ttarget=next(iter(train_data))
    toutput = cmodel(tdata)
    assert list(toutput.shape) ==[train_data.batch_size,512]
    assert list((toutput - ttarget).shape)==[train_data.batch_size,512]
    #
    mseloss = nn.MSELoss(reduction='none')
    poisloss = nn.PoissonNLLLoss(log_input=True)
    poislossnr = nn.PoissonNLLLoss(log_input=True,reduction='none')
    #
    def mseregcriterion(x,y,cliptens):
        return mseloss(x[cliptens],y[cliptens]).mean()
    # #x,y=output,target
    regcriterion=mseregcriterion
    cliptens = train_data.orfclip(tdata[0])
    mseregcriterion(toutput, ttarget, cliptens)
    #
    lr=0.001
    if 'esm' in datatouse:
        print('using esm')
        lr = 0.003
    #
    print('learning rate: {:3f}'.format(lr))
    # optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
    optimizer = torch.optim.Adam(cmodel.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
    train_data.batch_size=32
    val_data.batch_size=32
    print('batch size: {:3f}'.format(train_data.batch_size))


batchdata = next(iter(train_data))
batch=0

# fakefunk = ten()
# F.linear(batchdata[0])
# batchdata[0]
def get_rdata_cor(target, output, data, train_data):
    datacor = round(scipy.stats.pearsonr(
                target[train_data.orfclip(data[0],lclip,rclip)],
                output[train_data.orfclip(data[0],lclip,rclip)].detach()
    )[0],3)
    return datacor

batch = next(iter(val_data))

def get_val_cor(val_data,cmodel):
    val_data.batchshuffle=False
    cmodel.eval()
    with torch.no_grad():
        preds = [cmodel(b[0]).detach()[val_data.orfclip(b[0][0],lclip,rclip)] for b in val_data]
        targets = [b[1][val_data.orfclip(b[0][0],lclip,rclip)] for b in val_data]
    datacor = round(scipy.stats.pearsonr(
                torch.cat(preds),
                torch.cat(targets)
    )[0],4)
    cmodel.train()
    return datacor


def train():
    datacors = []
    valcors = []
    cmodel.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    bptt = train_data.batch_size
    for batch, batchdata in enumerate(train_data):
    # for batch in range(0,1):
        data,target = batchdata
        optimizer.zero_grad()
        output = cmodel(data)
        cliptens = train_data.orfclip(data[0])
        loss = regcriterion(output, target, cliptens)
        # ipdb.set_trace()
        if torch.isnan(loss): ipdb.set_trace()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(cmodel.parameters(), 0.5)
        # weightsold=cmodel.conv.weight.detach().numpy().copy() 
        #
        total_loss += loss.item()
        log_interval = 10
        # tweights = cmodel.conv.weight.detach().numpy()
        # tgrad = cmodel.conv.weight.grad
        # tchange = tweights - weightsold
        if batch % log_interval == 0 and batch > 0:
            # print('weights 0,2:'+'\n'+str(tweights[0,0,0])+'\n'+str(tweights[0,3,0]))
            # print('grads 0,3:'+'\n'+str(tgrad[0,0,0])+'\n'+str(tgrad[0,3,0]))
            # print('change 0,3:'+'\n'+str(tchange[0,0,0])+'\n'+str(tchange[0,3,0]))
            # print(tweights.grad.abs().mean())
            datacor = get_rdata_cor(target, output, data, train_data)
            print(f'batch: {batch}/{len(train_data) // bptt}')
            print(f'Correlation with training set: {round(datacor,4)}')
            vcor = get_val_cor(val_data, cmodel)
            print(f'Correlation with validation set: {round(vcor,4)}')
            datacors.append(datacor)
            valcors.append(vcor)
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if print_batch_loss: 
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, cur_loss))

            total_loss = 0
            start_time = time.time()
        optimizer.step()
    return datacors,valcors


# def evaluate(eval_model, data_source):
#     eval_model.eval() # Turn on the evaluation mode
#     total_loss = 0.
#     with torch.no_grad():
#         for i, batch in enumerate(data_source):
#             edata, targets = batch
#             output = eval_model(edata)
#             # output_flat = output.view(-1, ntokens)
#             output_flat = output.flatten
#             total_loss += len(edata) * regcriterion(output, targets).item()
#     return total_loss / (len(data_source) - 1)

print('training...')


def save_losses(train_losses,val_losses):
    cor_loss_df = pd.DataFrame(list(zip([v for sub in train_losses for v in sub],
    [v for sub in val_losses for v in sub])))
    epochbatches = (cor_loss_df.shape[0]//3)
    epoch = [[e] * epochbatches for e in range(3) ]
    epoch = [e for sub in epoch for e in sub]
    cor_loss_df['epoch'] = epoch
    cor_loss_df.columns = ['training','validation','epoch']
    cor_loss_df.to_csv('ix_tr_norm_trans_losses.tsv',sep='\t',index=False)

def train_model():
    val_losses=[]
    train_losses=[]
    best_val_loss= np.inf
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # print('first three weights:')
        # print(cmodel.conv.weight[0,0:3,0])
        # val_loss = evaluate(cmodel, val_data)
        # val_losses.append(val_loss)
        # print('-' * 89)
        # print('| start of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
        #       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                  val_loss, val_loss))
        print(f'epoch {epoch}/{epochs}')
        tr,val = train()
        val_losses.append(val)
        train_losses.append(tr)
        # print('-' * 89)

        # print(np.array(val_losses))

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = cmodel

        scheduler.step()

    save_losses(train_losses,val_losses)

train_model()

import itertools as it
rdata.usedata=train_data.usedata
def get_elongs(rdata, cmodel):
    rdata.usedata=datatouse
    rdata.batchshuffle=False
    next(iter(rdata))[0][0].shape
    cmodel.eval()
    with torch.no_grad():
        preds = [cmodel(b[0]).detach() for 
            b in rdata]
    preds = torch.cat(preds,axis=0)
    orfclipten = [val_data.orfclip(b[0][0],0,0) for b in rdata]
    orfclipten = torch.cat(orfclipten,axis=0)
    orfmeans = [preds[i][orfclipten[i]].mean().item() for 
        i in range(preds.shape[0])]
    txtdensity(np.array(orfmeans))
    cmodel.train()
    txtplot(preds[1])

    nms = list(rdata.gene2num.index)
    nms[1]
    rdata.batchshuffle=True
    return preds,orfmeans,nms

predslist = get_elongs(rdata, cmodel)

torch.save(predslist,args.o + '_ixnos_predict.pt')

elongdf = pd.DataFrame([pd.Series(predslist[1]),pd.Series(predslist[2])]).transpose()
elongdf.columns=['elong','tr_id']
elongdf.to_csv('trans_ix_elongs.csv')

################################################################################
########Test if it learns codon strengths the way I think it should
################################################################################
    
if False:
    n_cods=65
    fakecoddata=torch.diag(torch.FloatTensor([1]*n_cods)).reshape([1,n_cods,n_cods])
    otherdata = torch.zeros([1,tdata[0].shape[1]-n_cods,n_cods])
    fakedata = torch.cat([fakecoddata,otherdata],axis=1)




    # #okay so the fake data works, same as reading weights
    #
    if regcriterion is mseregcriterion:
        testoutput = cmodel((fakedata,tdata[1][0]))
        print('mse loss')
        txtplot(codstrengths,testoutput[:,1:])
    else:
        print('poisson loss')
        testoutput = cmodel((fakedata,tdata[1][0]))
        txtplot(codstrengths,testoutput[:,1:].exp())

    tdata[0].shape
    tdata,ttarget=next(iter(test_data))
    toutput = cmodel(tdata)


    txtplot(toutput[0])
    txtplot(toutput.sum(axis=0))
    txtplot(toutput.sum(axis=0)[0:10])

    losvallist.append(val_losses)
