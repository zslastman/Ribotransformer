import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

#
fakecodonseqs = torch.LongTensor(2**9,6000).random_(0,64)
fakecodonseqs1h = nn.functional.one_hot(fakecodonseqs)
#
codstrengths = torch.FloatTensor(range(0,64))
#
fakesignal = fakecodonseqs1h.float().matmul(codstrengths)
assert list(fakesignal.shape) == [512,6000]
assert list(fakecodonseqs.shape) == [512,6000]
assert list(fakecodonseqs1h.shape) == [512,6000,64]
fakecodonseqs1h = fakecodonseqs1h.transpose(1,0)
fakecodonseqs1h = fakecodonseqs1h.transpose(1,2)
assert list(fakecodonseqs1h.shape) == [6000,64,512]
fakecodonseqs1h = fakecodonseqs1h.float()
nn.Conv1d(64,1,1,padding=0)(fakecodonseqs1h)
# fcods_onehot = torch.BoolTensor(100,64,10^9)
# y_onehot.zero_()
# y_onehot.scatter_(1, y, 1)

#https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #zero vector of dim max_len,embdim
        pe = torch.zeros(max_len, d_model)
        #sequential double vector from 0 to max length of dim max_len, reshape to max_len,1 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #make a sequence in step 2 from 0 to embdim, so length embdim/2. 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #so every even one gets this
        pe[:, 0::2] = torch.sin(position * div_term)
        #every odd one gets this
        pe[:, 1::2] = torch.cos(position * div_term)
        #now take pe which is max_len,embdim(d_model), make it 1,max_len,embdim(d_model), then make it max_len,1,embdim(d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #here, to our len,batch,embdim vector, we add a len,1,embdim vector
        #this thing looks non identical across teh 200 dimensions 
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

ninp=64
#ninp is the embedding dimension
class Convmodel(nn.Module):
    def __init__(self, ninp, out, k, dropout=0.5):
        super(Convmodel, self).__init__()

        assert k%2 ==1

        self.conv = nn.Conv1d(ninp, out, k, padding=int(k/2))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.conv.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.conv(src)

        return output

lmodel = Convmodel(64, 1, 1).to(device)
lmodel(fakecodonseqs1h[0:10,:,:])

#ninp is the embedding dimension
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        #define individual layers as taking in ninp and outputing nhid, I think...
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        #now 
        # self.decoder = nn.Linear(ninp, ntoken)
        self.decoder = nn.Linear(ninp, 1)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        #make a len,batch,embdim tensor from the len,batch tensor
        src = self.encoder(src) * math.sqrt(self.ninp)
        #postiionally encode - leaves it the same dimensions
        src = self.pos_encoder(src)
        #this function  
        output = self.transformer_encoder(src, src_mask)
        #finally use this linear layer whose output is a linear function from 200 to nTokens (like 29k)
        #So the output of len,batch,embdim goes to len,batch,nTokens
        output = self.decoder(output)
        #with a linear layer we can drop the final dim
        output = output.squeeze(2)

        return output


import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

train_regtarget = make_regression_target_lag(train_data,0.2)
val_target = make_regression_target_lag(val_data, 0.2)
test_regtarget = make_regression_target_lag(test_data, 0.2)



def make_regression_target_lag(data,lam):
    ddata = torch.log1p(data.double())
    # ddata.shape
    # target = torch.zeros( torch.Size(list(data.shape)+[1]))
    target = torch.zeros(data.shape)
    for i in reversed(range(data.shape[0])):
        for j in range(data.shape[1]):
            target[i,j] += ddata[i,j] 
            target[i-1,j] += target[i,j] * lam
    for j in range(data.shape[1]): target[i-1,j] -= target[i,j] * lam
    return target
# ddata[:,0]
# target[:,0]
#right shape
#make_regression_target(data).shape
#data.shape

#source=train_data
bptt = 350
def reg_get_batch(source, regdata_source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = regdata_source[i:i+seq_len]
    return data, target


ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
regcriterion = nn.MSELoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)



import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # data, targets = get_batch(train_data, i)
        data,regtargets = reg_get_batch(train_data, train_regdata, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            #make the mask
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        #I've confirmed taht manually just doing log1p gets me a much lower loss
        loss = regcriterion(output, regtargets)
        # loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

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

# eval_model=model
# data_source=val_data
# target_source=val_target
def evaluate(eval_model, data_source, target_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = reg_get_batch(data_source,target_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            # output_flat = output.view(-1, ntokens)
            output_flat = output.view(-1, 1)
            output_flat = output_flat.squeeze(1)
            targets = targets.view(-1)
            total_loss += len(data) * regcriterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data, val_target)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()



test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)



################################################################################
########So the above works, how can I make it into a regressor
################################################################################
    

