import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotext as plx
import numpy as np
import random
import ipdb
import time

#https://pytorch.org/tutorials/beginner/transformer_tutorial.html

datalist = [codons]

if(USE_SEQ_FEATURES):
    datalist = datalist + [seqfeats]
if(GET_ESM_TOKENS and USE_ESM_TOKENS):
    datalist = datalist + [esmtensornorm]

combdata = torch.cat(datalist,axis=1)

n_feats = combdata.shape[1]
n_traingenes = combdata.shape[0]
print('Got '+str(n_feats)+' features for '+str(n_traingenes))


train_data, train_target, train_offsets = combdata[traininds], ribosignal[traininds], ribodens[traininds]
val_data, val_target, val_offsets = combdata[valinds], ribosignal[valinds], ribodens[valinds]
test_data, test_target, test_offsets = combdata[testinds], ribosignal[testinds], ribodens[testinds]

print('defining...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#ninp is the embedding dimension
class Convmodel(nn.Module):
    def __init__(self, ninp, out, k, dropout=0.5):
        super(Convmodel, self).__init__()

        assert k%2 ==1

        self.conv = nn.Conv1d(ninp, out, k, padding=int(k/2))

        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.conv.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.conv(src)
        output = output 
        return output

kernalsize=1

n_feats = train_data.shape[1]
n_traingenes = train_data.shape[0]
print('Using '+str(n_feats)+' features for '+str(n_traingenes)+' genes with a convolution window size of '+str(kernalsize))


cmodel = Convmodel(n_feats, 1, kernalsize).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert list(cmodel(train_data[0:10,:,:].to(device)).shape) ==[10,1,512]

mseloss = nn.MSELoss(reduction='none')
poisloss = nn.PoissonNLLLoss(log_input=True)

def regcriterion(x,y):
    return mseloss(x,y).mean()

# def regcriterion(x,y):
#     return poisloss(x,y).mean()

# regcriterion = nn.PoissonNLLLoss(log_input=True)
print('learning rate: {:3f}'.format(lr))

optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

i=0

trainloss = [] 

def train():
    cmodel.train() # Turn on the train mode
    total_loss = 0.
    total_lossout=0.
    start_time = time.time()

    tdatasize = train_data.shape[0]
    shuffinds = np.random.permutation(tdatasize)
    train_data_shuff=train_data[shuffinds],
    train_target_shuff=train_target[shuffinds],
    train_offsets_shuff=train_offsets[shuffinds],

    i=0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, regtargets, offsets = cget_batch(
            train_data[shuffinds],
            train_target[shuffinds],
            train_offsets[shuffinds],
            i)
        optimizer.zero_grad()
        output = cmodel(data)
        loss = regcriterion(output, offsets, regtargets)
        # ipdb.set_trace()
        if torch.isnan(loss): ipdb.set_trace()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(cmodel.parameters(), 0.5)
        optimizer.step()
        #
        total_loss += loss.item()
        if (i==0): total_lossout = loss.item()
        log_interval = 10
        # tweights = cmodel.conv.weight
        # print('weights 0,64:'+'\n'+str(tweights[0,0,0])+'\n'+str(tweights[0,63,0]))
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    print('tloss: '+str(total_loss/ log_interval))
    return total_loss/ log_interval


eval_model = cmodel
data_source = val_data
targetsource=target_source = val_target
offsets_source = val_offsets

def evaluate(eval_model, data_source, target_source, offsets_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets, offsets = cget_batch(data_source,target_source, offsets_source, i)
            output = eval_model(data)
            # output_flat = output.view(-1, ntokens)
            output_flat = output.flatten
            total_loss += len(data) * regcriterion(output+offsets, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
best_model = None
epoch=1



print('training...')


trainlosslist = [] 
vallosslist = [] 

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    val_loss = evaluate(cmodel, val_data, val_target, val_offsets)
    print('-' * 89)
    print('| start of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    tloss = train()
    trainlosslist.append(tloss)
    print('-' * 89)
    val_loss = evaluate(cmodel, val_data, val_target, val_offsets)
    vallosslist.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = cmodel

    scheduler.step()

def make_test_plot_data(n_cods,combdata):
    n_rest = combdata.shape[1] -  n_cods
    codtensor = torch.diag(torch.FloatTensor([1]*n_cods))
    codtensor = codtensor.reshape([1, n_cods,n_cods])
    resttensor = torch.zeros([n_rest,n_cods]).reshape(1,n_rest,n_cods)
    test_plot_data = torch.cat([codtensor,resttensor],axis=1)
    test_plot_data = test_plot_data.to(device)
    return test_plot_data  

test_plot_data = make_test_plot_data(n_cods,combdata)

def testplot():
    plx.clear_plot()
    plx.scatter(
            codstrengths.cpu().exp().numpy().flatten()[1:],
            cmodel(test_plot_data).cpu().detach().numpy().flatten()[1:], 
            rows = 17, cols = 70)
    plx.show()

testplot()


test_loss = evaluate(best_model, test_data, test_target, test_offsets)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# #force the weights to the values we think they should be
# cmodel.conv.weight[:,1:,:] = codstrengths[1:].reshape([1,n_cods-1,1])
# val_loss = evaluate(cmodel, val_data, val_target, val_offsets)
# print('|valid loss {:5.2f} | '.format(val_loss))



if USE_ESM_TOKENS:
    trainlosslist_tok = trainlosslist
    vallosslist_tok = vallosslist
else :
    trainlosslist = trainlosslist
    vallosslist_tok = vallosslist

################################################################################
########So the above works, how can I make it into a regressor
################################################################################
    

