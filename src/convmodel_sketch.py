import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


n_cods=64
n_genes = 6000
n_len = 512
#
fakecodonseqs = torch.LongTensor(n_len,n_genes).random_(0,n_cods)
fakecodonseqs1h = nn.functional.one_hot(fakecodonseqs)
#
codstrengths = torch.FloatTensor(range(0,n_cods))
#
fakesignal = fakecodonseqs1h.float().matmul(codstrengths)
fakesignal = fakesignal.transpose(1,0)
assert list(fakesignal.shape) == [n_genes,n_len]
assert list(fakecodonseqs.shape) == [n_len,n_genes]
assert list(fakecodonseqs1h.shape) == [n_len,n_genes,n_cods]
fakecodonseqs1h = fakecodonseqs1h.transpose(1,0)
fakecodonseqs1h = fakecodonseqs1h.transpose(1,2)
assert list(fakecodonseqs1h.shape) == [n_genes,n_cods,n_len]
fakecodonseqs1h = fakecodonseqs1h.float()
nn.Conv1d(n_cods,1,1,padding=0)(fakecodonseqs1h)
# fcods_onehot = torch.BoolTensor(100,64,10^9)
# y_onehot.zero_()
# y_onehot.scatter_(1, y, 1)
def cget_batch(source, targetsource, i, bptt=100):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = targetsource[i:i+seq_len]
    return data, target

assert list(cget_batch(fakecodonseqs1h, fakesignal, 1)[1].shape)==[100,512]
traininds = range(1,5000)
testinds = range(5000,5500)
valinds = range(5500,6000)
train_data, train_target = fakecodonseqs1h[traininds],fakesignal[traininds]
val_data, val_target = fakecodonseqs1h[valinds],fakesignal[valinds]
test_data, test_target = fakecodonseqs1h[testinds],fakesignal[testinds]


model.add(Conv1D(40, 1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(40, 1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(40, 21, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))
# cget_batch(train_data, fakesignal, i)[1].shape


#https://pytorch.org/tutorials/beginner/transformer_tutorial.html

#ninp is the embedding dimension
class Convmodel(nn.Module):
    def __init__(self,inplen, d_inp, k, dropout=0.5):
        super(Convmodel, self).__init__()

        assert k%2 ==1
        
        self.layers = [
            nn.Conv1d(d_inp, 40, 1, padding=int(k/2)),
            nn.Conv1d(  40, 40, 1, padding=int(k/2)),
            nn.Conv1d(  40, 40, 21, padding=int(k/2)),
            nn.Flatten(),
            nn.Linear(40*inplen, 20),
            nn.Linear(20, 1),
        ]
        # dropouts = [
        #     0.2,
        #     0.2,
        #     0.2,
        # ]

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.conv.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.conv(src)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernalsize=1
cmodel = Convmodel(n_cods, kernalsize).to(device)
assert list(cmodel(fakecodonseqs1h[0:10,:,:]).shape) ==[10,1,512]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

regcriterion = nn.MSELoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(cmodel.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

bptt=100



train_data=fakecodonseqs1h
train_regdata=fakesignal
import time
def train():
    cmodel.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, regtargets = cget_batch(train_data, train_regdata, i)
        optimizer.zero_grad()
        output = cmodel(data)
        loss = regcriterion(output.squeeze(1), regtargets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cmodel.parameters(), 0.5)
        optimizer.step()
        #
        total_loss += loss.item()
        log_interval = 200
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

val_data=fakecodonseqs1h[5000:5100]
val_target=fakesignal[5000:5100]

eval_model=cmodel
data_source=val_data
target_source=val_target

def evaluate(eval_model, data_source, target_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = cget_batch(data_source,target_source, i)
            output = eval_model(data)
            # output_flat = output.view(-1, ntokens)
            output_flat = output.flatten
            total_loss += len(data) * regcriterion(output.flatten(), targets.flatten()).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None
epoch=1


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(cmodel, val_data, val_target)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = cmodel

    scheduler.step()



test_loss = evaluate(best_model, test_data, test_target)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)



################################################################################
########So the above works, how can I make it into a regressor
################################################################################
    

