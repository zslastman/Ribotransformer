################################################################################
########Very simple convolutional model, this works with fake data really well
########Also learns something with the real yeast data
################################################################################
    
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

# original code from.https://pytorch.org/tutorials/beginner/transformer_tutorial.html

print('defining...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#ninp is the embedding dimension
class RiboModel(nn.Module):
    def flipBatch(data, lengths):
        assert data.shape[0] == len(lengths), "Dimension Mismatch!"
        for i in range(data.shape[0]):
            data[i,:lengths[i]] = data[i,:lengths[i]].flip(dims=[0])

        return data

    def __init__(self, ninp, out, k, dropout=0.5):
        super(RiboModel, self).__init__()

        assert k%2 ==1

        n_lstm=2

        self.conv = nn.Conv1d(ninp, out, k, padding=int(k/2))

        self.lstm = nn.LSTM(ninp + 1, n_lstm, bidirectional=True)

        self.convfinal = nn.Conv1d(n_lstm*2, 1, 1, padding=int(1/2))

        self.init_weights(ninp)

    def init_weights(self,ninp):
        initrange = 0.1/ninp
        self.conv.weight.data.uniform_(-initrange, initrange)

    # src_offset = tdata
    def forward(self, src_offset):
        src,offset = src_offset

        output = self.conv(src)
        
        output = torch.cat([src,output],axis=1)

        # output = src

        output_lstm,hidden = self.lstm(output.permute([2,0,1]))
        output_lstm = output_lstm.permute([1,2,0])

        output = self.convfinal(output_lstm)

        output = output + offset.log()
        # output = output * offset
        return output.squeeze(1)

train_data.batch_size=100
test_data.batch_size=100
val_data.batch_size=100

rdata.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datadim = train_data.datadim()
kernalsize=1
cmodel = RiboModel(datadim, 1, kernalsize).to(rdata.device)

#test the model
# assert rdata.datadim() is 80
# assert val_data.datadim() is 80
tdata,ttarget=next(iter(train_data))
toutput = cmodel(tdata)
assert list(toutput.shape) ==[train_data.batch_size,512]
assert list((toutput - ttarget).shape)==[train_data.batch_size,512]

mseloss = nn.MSELoss(reduction='none')
poisloss = nn.PoissonNLLLoss(log_input=True)
poislossnr = nn.PoissonNLLLoss(log_input=True,reduction='none')

def mselossfun(x,y):
    return mseloss(x,y)
#x,y=output,target
def poisslossfun(x,y):
    eps = 1/y.mean(axis=1).reshape([-1,1])
    return poisloss(x,y+eps)
    # poislossnr(x,y+eps)
regcriterion = mselossfun
regcriterion = poisslossfun
# ############

test_data.batch_size=10000
tdata,ttarget=next(iter(test_data))
src_offset=tdata
toutput = cmodel(tdata)
regcriterion(toutput,ttarget)

# tdata,ttarget=next(iter(train_data))
# tdata,y=tdata,ttarget
# x=cmodel(tdata)
# eps = 1/y.mean(axis=1).reshape([-1,1])
# testloss = poislossnr(x,y+eps)
# infgeneinds = np.where(testloss.sum(axis=1).detach().numpy()==np.inf)
# assert len(infgeneinds[0])>0,'no genes with infinit loss'

# tgene = infgeneinds[0][0]
# maxind = y[tgene].argmax()
# y[tgene,maxind]#some very high number
# x[tgene,maxind]

# regcriterion(cmodel(tdata),ttarget)
# cmodel.conv.weight[0,0,0]= -3

# np.where(tdata[0][0,:,0].numpy())

# In [1275]: cmodel.conv.weight.shape
# Out[1275]: torch.Size([1, 80, 1])
# # regcriterion = nn.PoissonNLLLoss(log_input=True)
# #I should be able to influence the output on the first place by changing
# #weight 54

# cmodel(tdata)[0,0:2]
# 4.26
# #now if I change the weight
# cmodel.conv.weight[0,64,0]=4
# cmodel(tdata)[0,0:2]
# #true
# #maybe also with 66
# cmodel.conv.weight[0,64,0]=-2
# cmodel(tdata)[0,0:2]

# regcriterion(cmodel(tdata),ttarget)

# #true

#specific instance where loss is infinite(and generally just really high)

# ############

lr = 0.005
print('learning rate: {:3f}'.format(lr))
optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


batchdata = next(iter(train_data))
batch=0
def train():
    cmodel.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    bptt = train_data.batch_size

    for batch, batchdata in enumerate(train_data):

        data,target = batchdata
        optimizer.zero_grad()
        output = cmodel(data)
        loss = regcriterion(output, target)
        # ipdb.set_trace()
        if torch.isnan(loss): ipdb.set_trace()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(cmodel.parameters(), 0.5)
        optimizer.step()
        #
        total_loss += loss.item()
        cur_loss = loss.item()
        log_interval = 10

        # print('weights 0,64:'+'\n'+str(tweights[0,0,0])+'\n'+str(tweights[0,63,0]))
        if batch % log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} '.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            start_time = time.time()
##
epoch=1
# train()

eval_model=cmodel
data_source=val_data
i=0
batch = next(iter(data_source))

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    n=0
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            edata, targets = batch
            output = eval_model(edata)
            # output_flat = output.view(-1, ntokens)
            # output_flat = output.flatten
            total_loss += regcriterion(output, targets).item()
            n+=1
    return total_loss / n


best_val_loss = float("inf")
best_model = None
epochs=10
epoch=1

print('training...')

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    print('first three weights:')
    print(cmodel.conv.weight[0,0:3,0])
    val_loss = evaluate(cmodel, val_data)
    print('-' * 89)
    print('| start of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          ''.format(epoch, (time.time() - epoch_start_time),
                                     val_loss))
    train()
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = cmodel

    scheduler.step()

################################################################################
########Test if it learns codon strengths the way I think it should
################################################################################
    
n_cods=65
fakecoddata=torch.diag(torch.FloatTensor([1]*n_cods)).reshape([1,n_cods,n_cods])
otherdata = torch.zeros([1,tdata[0].shape[1]-n_cods,n_cods])
fakedata = torch.cat([fakecoddata,otherdata],axis=1)

# #okay so the fake data works, same as reading weights
# testoutput = cmodel((fakedata,tdata[1][0]))
# plx.clear_plot()
# plx.scatter(
#         cmodel.conv.weight.detach().numpy().flatten()[1:],
#         testoutput.cpu().detach().numpy().flatten()[1:],
#         rows = 17, cols = 70)
# plx.show()

#
testoutput = cmodel((fakedata,tdata[1][0]))

plx.clear_plot()
plx.scatter(
        codstrengths.cpu().detach().numpy().flatten(),
        testoutput.exp().cpu().detach().numpy().flatten()[1:],
        rows = 17, cols = 70)
plx.show()

################################################################################
########Positional effect learned?
################################################################################
def txtplot(v):
    if 'torch' in str(type(v)): v = v.detach().numpy()
    ro.globalenv['rsignal'] =  v
    r('library(txtplot);txtplot(rsignal)')
    
test_data.batch_size=1000
valbatchdata = next(iter(test_data))[0]
val_output_profile = cmodel(valbatchdata).mean(axis=0)

valbatchdata[0].shape
# val_output_profile.max()
txtplot(val_output_profile)
txtplot(val_output_profile[0:20])

##also test if it understands the start and end of the cds
valbatchdata[0][:,:,300:]=0
valbatchdata[0][:,0,300:]=1

val_output_profile_endalt = cmodel(valbatchdata).mean(axis=0)

plx.clear_plot()
plx.scatter(
        range(len(val_output_profile)),
        val_output_profile_endalt.detach().numpy(),
        rows = 17, cols = 70)
plx.show()


#YESSS

#what if I fake rdata to look the way I think it should?

################################################################################
########Is everything working right in a totally fake scenario?
################################################################################
##Yes, with poisson loss, and my single example things update in the right direction

fakesignal = ten(range(0,65)).reshape([1,65])
fakesrc = fakedata,tdata[1][0]**0

#fake data into model
cmodel(fakesrc)
#loss fun on fake signal

cmodel = Convmodel(datadim, 1, kernalsize).to(rdata.device)
lr = 0.005
optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

cmodel.train()
loss = regcriterion(cmodel(fakesrc),fakesignal)
loss.backward()
loss
optimizer.step()
regcriterion(cmodel(fakesrc),fakesignal)

cmodel.conv.weight.exp()
#this gets us increasingly negative gradients

if True:
    wts = cmodel.conv.weight

    wts[0,0:3,0]
    wts[0,62:65,0]
    wts.grad[0,0:3,0]
    wts.grad[0,62:65,0]
    #save old values
    oldweights = wts.detach().numpy()

    #
    optimizer.step()
    #now get again
    newweights = cmodel.conv.weight.detach().numpy()
    oldweights==newweights



# plx.clear_plot()
# plx.scatter(
#         codstrengths.cpu().numpy().flatten()[1:],
#         cmodel(torch.diag(torch.FloatTensor([1]*65)).reshape([1,65,65])).detach().numpy().flatten()[1:], 
#         rows = 17, cols = 70)
# plx.show()

# #force the weights to the values we think they should be
# val_loss = evaluate(cmodel, val_data)
# print('|valid loss {:5.2f} | '.format(val_loss))
# cmodel.conv.weight[:,1:,:] = codstrengths.reshape([1,64,1]).log()
# val_loss = evaluate(cmodel, val_data)
# print('|valid loss {:5.2f} | '.format(val_loss))