

if True:
    class StructNet(nn.Module):
        def __init__(self, n_feats, n_hid, finalmodelweights=None):
            super(StructNet, self).__init__()
            self.h1 = nn.Linear(n_feats, n_hid)  # 5*5 from image dimension
            self.h2 = nn.Linear(n_hid, n_hid)  # 5*5 from image dimension
            self.h3 = nn.Linear(n_hid, n_hid)  # 5*5 from image dimension
            self.h4 = nn.Linear(n_hid, n_hid)  # 5*5 from image dimension
            self.out = nn.Linear(n_hid, 3)
            if not finalmodelweights is None:
                self.init_weights(n_hid, finalmodelweights)
        def init_weights(self, n_hid, finalmodelweights):
            self.h1.weight.data = torch.Tensor(finalmodelweights[0].T)
            self.h1.bias.data = torch.Tensor(finalmodelweights[1].T)
            self.out.weight.data = torch.Tensor(finalmodelweights[2].T).reshape([1,n_hid])
            self.out.bias.data = torch.Tensor(finalmodelweights[3].T).reshape([1])
        # self,x = mlp, X_tr
        def forward(self, x):
            x = F.relu(self.h1(x))
            x = F.relu(self.h2(x))
            x = F.relu(self.h3(x))
            x = F.relu(self.h4(x))
            x = self.out(x)
            return x
    # 
    acgt=np.array(['A','C','G','T'])
    # i=0
    # np.where(struct_X_tr[:,i:i+4])[1].shape
    # i=1
    # np.where(struct_X_tr[:,i:i+4])[1].shape
    # np.stack([acgt[np.where(struct_X_tr[:,i:i+4])[1]].shape for i in range(0,10)])
    # 
    offset=5
    struct_X_tr = X_tr[:-offset,640:-3]
    struct_y_tr = X_tr[offset:,-3:]
    struct_X_val = X_val[:-offset,640:-3]
    struct_y_val = X_val[offset:,-3:]
    #
    trainset = Dataset(torch.Tensor(struct_X_tr),torch.Tensor(struct_y_tr))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                             shuffle=True, num_workers=1)
    i,tdata = next(enumerate(trainloader,0))
    #
    net = StructNet(struct_X_tr.shape[1],n_hid = 200)
    criterion = nn.MSELoss()
    assert net(tdata[0]).shape==tdata[1].shape
    criterion(net(tdata[0]),tdata[1])
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9,nesterov=True)
    lr_decay = 16
    lrlambda = lambda epoch: 1 / (1 + float(epoch) / lr_decay)
    # [lrlambda(i) for i in range(0,3)]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrlambda)

if True:
    #
    #
    for epoch in range(55):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = tdata
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            # print(net.h1.weight[0:3,0:3])
            # print(net.h1.weight.grad[0:3,0:3])
            running_loss += loss.item()
            if i % 20  == 20-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        scheduler.step()
        loss = criterion(net(torch.Tensor(struct_X_val)), torch.Tensor(struct_y_val))
        print(f'validation loss: {loss}')
        if epoch>0:
            change =  loss - oldvalloss
            print(f'val loss change: {change}')
        oldvalloss = loss
        print('mine vs actual:')

    # finalmodel.X_tr[0:dlen,0+(64*5):64+(64*5)]

def X_to_seq(X):
    alpha="ACGT"
    codons = [x+y+z for x in alpha for y in alpha for z in alpha]
    cod2id = {codon:idx for idx, codon in enumerate(codons)}
    id2cod = dict(zip(cod2id.values(),cod2id.keys()))
    assert X.shape[1]==4
    seqfromdat = ''.join([id2cod[cod] for cod in np.where(X)[1]])
    return(seqfromdat)


i=0
Xallseq = X_to_seq(X_tr[:,640:640+4])
import re
for i in range(0,X_tr.shape[0]):
    s = X_to_seq(X_tr[i:][:10,640:640+4])
    match = re.finditer(s,Xallseq)

minds = [r.start() for r in match]
#okay so we have two matches
assert Xallseq[minds[0]:][:30]==Xallseq[minds[1]:][:30]
assert Xallseq[minds[0]+1:][:30]==Xallseq[minds[1]+1:][:30]
assert Xallseq[minds[0]+2:][:30]==Xallseq[minds[1]+2:][:30]
#extends for 3bp

X_tr[:,-3:][int(minds[0]//3)-2:][:6]
X_tr[:,-3:][int(minds[1]//3)-2:][:6]

#the feature mats line up
assert (X_tr[:,640:640+4][int(minds[0]//3):][:10]==X_tr[:,640:640+4][int(minds[1]//3):][:10]).all()

#so where does the energy line up?]
X_tr[:,-3:][int(minds[0]//3)+5:][:10]==X_tr[:,-3:][int(minds[1]//3)+5:][:10]
X_tr[:,-3:][int(minds[0]//3)-5:][:10]
X_tr[:,-3:][int(minds[1]//3)-5:][:10]

#Let's see if we can find an identical sequence local, and then 
#from the file
#/fast/AG_Ohler/dharnet/Ribotransformer/iXnos/structure_data/scer.13cds10.windows.30len.fold
# ACGGUUGGUGUUUCGUAAUUUGAAUGUUGG
# ((((........)))).............. ( -1.80)
# >YAL008W        24
# CGGUUGGUGUUUCGUAAUUUGAAUGUUGGG
# ..........((((.....))))....... ( -0.30)
# >YAL008W        25
# GGUUGGUGUUUCGUAAUUUGAAUGUUGGGA
# ..........((.((((......)))).)) ( -2.30)

#get the energies as a flat vector
flatvect = X_tr[:100,-3:][:100].reshape([-1])

#now find that seq above
np.isin(np.array([-4.4,-4.4,-4.4]),(flatvect))

sfile = '/fast/AG_Ohler/dharnet/Ribotransformer/iXnos/structure_data/scer.13cds10.windows.30len.fold'
with open(sfile) as myfile:
    head = [next(myfile) for x in range(3*100)]


seqs = [h.strip() for h in head[1::3]]
energieshead = [re.findall('-?\d\d*\.?\d*',h)[0] for h in head[2::3]]

def min_e(seq):
    import seqfold
    from seqfold import dg, dg_cache, fold, Struct
    seq = "GGGAGGTCGTTACATCTGGGTAACACCGGTACTGATCCGGTGACCTCCC"
    # folds = fold(seq)
    # min_e = np.array([f.e for f in folds]).min()
    min_e = np.array([f.e for f in folds]).min()
    return min_e


min_e_list = [min_e(s) for s in seqs]
min_e_list = [dg(s) for s in seqs]

stats.pearsonr(min_e_list,np.array(energieshead)))





((20-5)*3)
