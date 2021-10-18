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

for datatouse in (['codons'],['codons','esm'],['esm']):
    # datatouse = ['codons']
    train_data.usedata=datatouse
    val_data.usedata=datatouse
    test_data.usedata=datatouse
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

            # output = output + offset.log()
            # output = output * offset
            return output.squeeze(1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datadim = train_data.datadim()
    kernalsize=1
    cmodel = Convmodel(datadim, 1, kernalsize).to(device)

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



    lr=1
    if 'esm' in datatouse:
        lr = 0.003

    print('learning rate: {:3f}'.format(lr))
    optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9)


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
            weightsold=cmodel.conv.weight.detach().numpy().copy()
            optimizer.step()
            #
            total_loss += loss.item()
            log_interval = 10
            tweights = cmodel.conv.weight.detach().numpy()
            tgrad = cmodel.conv.weight.grad
            tchange = tweights - weightsold
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
    epochs=20
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

    txtplot(toutput.sum(axis=0))

    losvallist.append(val_losses)


#

omitn=5
losses2plot=np.concatenate([losvallist[i][omitn:] for i in range(len(losvallist))])
xcoords = np.array(list(range(0,len(losses2plot))))
txtplot(xcoords,losses2plot)

#what if I fake rdata to look the way I think it should?

# foo
# ################################################################################
# ########Is everything working right in a totally fake scenario?
# ################################################################################
# ##Yes, with poisson loss, and my single example things update in the right direction

# fakesignal = ten(range(0,65)).reshape([1,65])
# fakesrc = fakedata,tdata[1][0]**0

# #fake data into model
# cmodel(fakesrc)
# #loss fun on fake signal

# cmodel = Convmodel(datadim, 1, kernalsize).to(rdata.device)
# lr = 0.005
# optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# cmodel.train()
# loss = regcriterion(cmodel(fakesrc),fakesignal)
# loss.backward()
# loss
# optimizer.step()
# regcriterion(cmodel(fakesrc),fakesignal)

# cmodel.conv.weight.exp()
# #this gets us increasingly negative gradients

# if True:
#     wts = cmodel.conv.weight

#     wts[0,0:3,0]
#     wts[0,62:65,0]
#     wts.grad[0,0:3,0]
#     wts.grad[0,62:65,0]
#     #save old values
#     oldweights = wts.detach().numpy()

#     #
#     optimizer.step()
#     #now get again
#     newweights = cmodel.conv.weight.detach().numpy()
#     oldweights==newweights



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




