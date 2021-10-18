if True:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # import plotext as plx
    import numpy as np
    import random
    t = torch.Tensor

if True:
    n_cods = 64
    # n_cods=5
    n_genes = 6000
    # n_genes = 3
    n_len = 512
    #generate integer n_len,n_gene tensor showing codons for each gene
    fakecodonseqs = torch.LongTensor(n_len, n_genes).random_(0, n_cods)
    #now make it one hot, add a 
    codons = nn.functional.one_hot(fakecodonseqs)
    #
    codstrengths = 1.05**np.linspace(-1 - (60 / 2), 2 + (60 / 2), 4 + 60)
    codstrengths = torch.FloatTensor(codstrengths)
    #
    codsig = codons.float().matmul(codstrengths)

    starteffshape = np.concatenate(
        [1.5**(-np.linspace(0, 10, 10)), np.array([0] * (512 - 10))])
    assert starteffshape.shape == (n_len,)
    stopeffshape = np.concatenate(
        [np.array([0] * (512 - 10)), 1.5**(-np.linspace(0, 10, 10))])
    assert stopeffshape.shape == (n_len,)
    starteffstrength = torch.zeros(n_genes)
    starteffstrength[5000:5500]=t(range(10+0,10+500))
    stopeffstrength = torch.zeros(n_genes)
    stopeffstrength[4750:5250]=t(range(10+0,10+500))
    starteff = torch.mm(t(starteffshape).reshape([-1,1]),starteffstrength.reshape([1,-1]))
    stopeff = torch.mm(t(stopeffshape).reshape([-1,1]),stopeffstrength.reshape([1,-1]))
    #
    genetpmvect = 2 * (10**np.linspace(1, 4, n_genes)) / n_len
    genetpmvect = genetpmvect+10
    # genetpmvect = genetpmvect / genetpmvect
    genetpmvect = torch.FloatTensor(genetpmvect)
    #
    # n,n_genes codsig by (n_genes,n_genes)
    # tpmmatrix gets us a n,n_genes signal mat
    fakesignal = codsig * genetpmvect.reshape(1, n_genes)

    fakesignal = fakesignal + \
        starteff + \
        stopeff 

    fakesignal = torch.poisson(fakesignal)
    fakesignal = fakesignal.transpose(1, 0)



# b = t(range(1,25)).reshape([4,6])
if True:
    b=fakesignal
    #this intersperses our vector so it's got something once every 3
    track = torch.stack([b,b-b,b-b]).permute([1,2,0]).reshape([n_genes,n_len*3])
    c_in = 1
    k = 40+40+1
    mainoffset = 12
    cent=k//2
    shift = 2
    rl_props = t([1,4,2])
    cout = len(rl_props)
    m = torch.Tensor(np.zeros([cout,c_in,k]))
    shiftv = t([0,1,3,6,3,1,0])
    rlshiftm = np.matmul(shiftv.reshape([-1,1]),rl_props.reshape([1,-1]))
    assert list(rlshiftm.shape)==[len(shiftv),cout]
    rlshiftm = rlshiftm/rlshiftm.sum()
    for i_cout in range(cout):
        for i in range(len(shiftv)):
            vhalf = len(shiftv)//2
            m[i_cout,0,cent+mainoffset-vhalf+i] = rlshiftm[i,i_cout]#add a natural number, shift back by that amount
    # m[0,0,cent-1] = 1
    # m[0,0,cent+1] = 0.3
    # m[0,0,cent+2] = 0.7
    #this shifts the
    trackpad = F.pad(track, pad = (40,40,0,0))
    out = F.conv1d(trackpad.reshape([n_genes,1,-1]),weight=m,padding=int(k/2))
    #indeed this preserves
    track.sum(1)
    out.sum([1,2])
    out[0,0,:]
    out.shape

# m_inv = m.permute([1,0,2]).flip([2])
m_inv = m
m_inv[m_inv!=0] = m_inv[m_inv!=0] ** -1
# F.conv1d(out,weight=m_inv,padding=int(k/2)).sum()

tout = F.conv_transpose1d(out[:100,:,:],weight=m_inv,padding=int(k/2))

assert (tout - trackpad[0:100]).max()

tout.shape
out[:100].shape
out[0].sum()
tout.sum()

outint = torch.poisson(out)
assert outint.sum()!=0




#ninp is the embedding dimension
class Autoenc(nn.Module):
    def __init__(self, ninp, out, k, dropout=0.5):
        super(Convmodel, self).__init__()

        assert k%2 ==1

        self.conv1 = nn.Conv1d(ninp, out, k, padding=int(k/2))
        self.conv2 = nn.Conv1d(out, n_inp, k, padding=int(k/2))
        self.linear1 = nn.Conv1d(ninp, out, k, padding=int(k/2))
        self.linear2 = nn.Conv1d(out, n_inp, k, padding=int(k/2))

        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.conv1.weight.data.uniform_(-initrange, initrange)
        self.conv2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        #go from genes,readlens,pos to genes,pos,summarydims
        activation = self.conv1(src)
        activation = F.relu(activation)
        #go from genes,pos to genes,summarydims
        activation = self.linear1(activation)
        activation = self.linear2

        output = self.conv2(activation)
        return output




myconv = nn.Conv1d()


# var_exists = 'test_target' in locals() or 'test_target' in globals()

# if not var_exists:

poseffects = True








    assert list(fakesignal.shape) == [n_genes, n_len]
    assert list(fakecodonseqs.shape) == [n_len, n_genes]
    assert list(codons.shape) == [n_len, n_genes, n_cods]
    codons = codons.transpose(1, 0)
    codons = codons.transpose(1, 2)
    assert list(codons.shape) == [n_genes, n_cods, n_len]
    codons = codons.float()
    nn.Conv1d(n_cods, 1, 1, padding=0)(codons)
    # fcods_onehot = torch.BoolTensor(100,64,10^9)
    # y_onehot.zero_()
    # y_onehot.scatter_(1, y, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TPMoffsets = torch.log(fakesignal.mean(axis=1)**-1)

# source, targetsource, offsetsource
# i = train_data, train_regdata, train_offsets, i

    def cget_batch(
            source, targetsource, offsetsource,
            i, bptt=100, device=device):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = targetsource[i:i + seq_len].unsqueeze(1)
        offset = offsetsource[i:i + seq_len].unsqueeze(1).unsqueeze(1)
        return data.to(device), target.to(device), offset.to(device)

    assert list(cget_batch(codons, fakesignal, TPMoffsets, 1)
                [0].shape) == [100, n_cods, n_len]
    assert list(cget_batch(codons, fakesignal, TPMoffsets, 1)
                [1].shape) == [100, 1, n_len]
    assert list(cget_batch(codons, fakesignal, TPMoffsets, 1)
                [2].shape) == [100, 1, 1]

    allinds = np.array(random.sample(range(0, n_genes), k=n_genes))

    traininds = allinds[0:(len(allinds) - 1000)]
    testinds = allinds[(len(allinds) - 1000):(len(allinds) - 500)]
    valinds = allinds[(len(allinds) - 500):]

    train_data, train_target = codons[traininds], fakesignal[traininds]
    val_data, val_target = codons[valinds], fakesignal[valinds]
    test_data, test_target = codons[testinds], fakesignal[testinds]

    # we add these to the output of our model, to normalize for TPM
    train_offsets = TPMoffsets[traininds]
    val_offsets = TPMoffsets[valinds]
    test_offsets = TPMoffsets[testinds]

# we add these to the output of our model, to normalize for TPM
(fakesignal * codons[:, 1, :]).shape
(TPMoffsets.unsqueeze(1)).exp().shape


# calculate average signal per codon
# just proving this works for when i do it with data
pcodstrengths = [((fakesignal) / (TPMoffsets.exp().unsqueeze(1)))
                 [codons[:, i, :] == 1].mean().item() for i in range(0, n_cods)]
pcodstrengths = torch.FloatTensor(pcodstrengths)
