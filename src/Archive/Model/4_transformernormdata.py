################################################################################
########Very simple convolutional model, this works with fake data really well
########Also learns something with the real yeast data
################################################################################

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

    ten = torch.Tensor
    print_batch_loss = False

    # original code from.https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    print('defining...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def norm_ribosig(data2norm):
        data2norm.ribosignal.shape

        data2norm.ribosignal -= data2norm.ribosignal.mean(axis=1).reshape([-1,1])
        data2norm.ribosignal /= data2norm.ribosignal.std(axis=1).reshape([-1,1])
        # data2norm.offset = None
        return data2norm

    train_data,val_data,test_data = map(norm_ribosig,(train_data,val_data,test_data))

    losvallist = []


if True:
    #https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    #this creates a function that gets a vector of len,1,d_model and adds it to a a tensor of e.g. len,n,d_model
    class PositionalEncoding(nn.Module):

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            d_model = 2*int(d_model/2)
            pe = torch.zeros(max_len, d_model)
            #5000,200
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            #5000,1
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            #shape 100
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            #pe.shape==5000,1,200
            self.register_buffer('pe', pe)

        def forward(self, x):
            x[:,:,:self.pe.size(2)] += self.pe[:x.size(0), :]
            return self.dropout(x)


    # for datatouse in (['codons'],['codons','esm'],['esm']):
    # for datatouse in (['codons']):
    datatouse = ['codons']
    train_data.usedata=datatouse
    val_data.usedata=datatouse
    test_data.usedata=datatouse

    #ninp is the embedding dimension
    #ninp,nhead,nhid,dropout = 65,10,10,0.5
    class TransformerModel(nn.Module):
        def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
            super(TransformerModel, self).__init__()
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(65, nhead, nhid, dropout)
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
            src[:,1:,:]=0
            # assert list(src.shape) == [300,65,512] 
            src = src.permute([2,0,1])
            src = self.pos_encoder(src)
            # output = self.conv(src)
            # output = output + offset.log()
            # output[src[:,0:1,:]==1]=0
            # output = output * offset
            output = self.transformer_encoder(src) 
            # assert list(output.shape) == (512,n_batch,n_feats)
            output = self.decoder(output)
            # assert list(output.shape) == (512,n_batch,1) 
            return output.squeeze(2).permute([1,0])
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadim = train_data.datadim()
    cmodel = TransformerModel(datadim, 5,20,5).to(device)
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
def mseregcriterion(x,y):
    return mseloss(x,y).mean()
# #x,y=output,target
regcriterion=mseregcriterion



lr=0.1
if 'esm' in datatouse:
    print('using esm')
    lr = 0.003

print('learning rate: {:3f}'.format(lr))
# optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
optimizer = torch.optim.Adam(cmodel.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)

batchdata = next(iter(train_data))
batch=0
def train():
    cmodel.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    bptt = train_data.batch_size

    # for batch, batchdata in enumerate(train_data):
    for batch in range(0,20):

        data,target = batchdata
        optimizer.zero_grad()
        output = cmodel(data)
        loss = regcriterion(output, target)
        # ipdb.set_trace()
        if torch.isnan(loss): ipdb.set_trace()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(cmodel.parameters(), 0.5)
        # weightsold=cmodel.conv.weight.detach().numpy().copy()
        optimizer.step()
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
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if print_batch_loss: print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, cur_loss))
            total_loss = 0
            start_time = time.time()

##
epoch=1
train()

eval_model=cmodel
data_source=val_data
i=0
batch = next(iter(data_source))

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.

    with torch.no_grad():
        for i, batch in enumerate(data_source):
            edata, targets = batch
            output = eval_model(edata)
            # output_flat = output.view(-1, ntokens)
            output_flat = output.flatten
            total_loss += len(edata) * regcriterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
best_model = None
epochs=10
epoch=1

print('training...')
val_losses=[]
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    # print('first three weights:')
    # print(cmodel.conv.weight[0,0:3,0])
    val_loss = evaluate(cmodel, val_data)
    val_losses.append(val_loss)
    # print('-' * 89)
    # print('| start of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
    #       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
    #                                  val_loss, val_loss))
    train()
    # print('-' * 89)

    txtplot(np.array(val_losses))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = cmodel

    scheduler.step()

if False:
    ################################################################################
    ########Test if it learns codon strengths the way I think it should
    ################################################################################
        
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

    tdata,ttarget=next(iter(test_data))
    toutput = cmodel(tdata)

    txtplot(toutput[0])
    txtplot(toutput.sum(axis=0))
    txtplot(toutput.sum(axis=0)[0:10])



