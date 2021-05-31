################################################################################
########Let's replicate their perceptron, see if I can pytorch something almost
########as good
################################################################################
ten=torch.Tensor

class Net(nn.Module):
    def __init__(self, n_feats, n_hid, finalmodelweights=None):
        super(Net, self).__init__()
        self.h1 = nn.Linear(n_feats, n_hid)  # 5*5 from image dimension
        self.out = nn.Linear(n_hid, 1)
        if not finalmodelweights is None:
            self.init_weights(n_hid, finalmodelweights)
    def init_weights(self, n_hid, finalmodelweights):
        self.h1.weight.data = torch.Tensor(finalmodelweights[0].T)
        self.h1.bias.data = torch.Tensor(finalmodelweights[1].T)
        self.out.weight.data = torch.Tensor(finalmodelweights[2].T).reshape([1,n_hid])
        self.out.bias.data = torch.Tensor(finalmodelweights[3].T).reshape([1])
    # self,x = mlp, X_tr
    def forward(self, x):
        hidden = torch.tanh(self.h1(x))
        out = F.relu(self.out(hidden))
        return out
        
n_feats,n_hid = X_tr.shape[1],200
mlp = Net(n_feats, n_hid, finalmodelweights)

samewout = mlp(torch.Tensor(X_tr))
# (samewout.detach().numpy() - y_tr_hat).mean()

txtplot(samewout,y_tr_hat)
stats.pearsonr(samewout.detach().numpy().flatten(),y_tr_hat.flatten())
print('iXnos vs trained:')
print(stats.pearsonr(samewout.detach().numpy().flatten(),y_tr_hat.flatten()))
print('mine vs actual:')
print(stats.pearsonr(samewout.detach().numpy().flatten(),y_tr.flatten()))
print('iXnos vs actual:')
print(stats.pearsonr(y_tr_hat.flatten(),y_tr.flatten()))

'loss function'
criterion = nn.MSELoss()
print(criterion(samewout,ten(y_tr).reshape(-1,1)))
print(criterion(ten(y_tr_hat),ten(y_tr)))

#Why is this different??
((y_tr_hat - y_tr)**2).mean()

#what if I set influence of the structure to 0?
mlp.h1.weight[:,-3:]= 0
nstruct_samewout = mlp(torch.Tensor(X_tr))
txtplot(samewout,nstruct_samewout)
print('mine vs actual:')
print(stats.pearsonr(nstruct_samewout.detach().numpy().flatten(),y_tr.flatten()))
#okay that makes it quite a lot worse

################################################################################
########Now let's see if I can train one that's as good
################################################################################

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y
  #
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
  #
  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.X[index]
        y = self.y[index]
        return X, y
#
trainset = Dataset(torch.Tensor(X_tr[:,:-3]),torch.Tensor(y_tr.reshape(-1,1)))
trainloader = trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,shuffle=True, num_workers=1)
i,data = next(enumerate(trainloader,0))
#
n_feats = data[0].shape[1]
net = Net(n_feats, n_hid)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9,nesterov=True)
lr_decay = 16
lrlambda = lambda epoch: 1 / (1 + float(epoch) / lr_decay)
# [lrlambda(i) for i in range(0,3)]
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrlambda)
#
def train(trainloader,net,criterion,optimizer,X_te,y_te):
    bestloss = 1e12
    for epoch in range(55):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # if i % 20  == 20-1:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                      # (epoch + 1, i + 1, running_loss / 20))
                # running_loss = 0.0
        scheduler.step()
        loss = criterion(net(torch.Tensor(X_te[:,:n_feats])), torch.Tensor(y_te).reshape(-1,1))
  
        myout = net(torch.Tensor(1.0*X_te))
        cor = stats.pearsonr(myout.detach().numpy().flatten(),y_te.flatten())[0]
        sys.stdout.write(f'Epoch: {epoch} mine vs actual:\tcor: {cor:.4f}\r')
        if loss < bestloss:
            bestloss = loss
            bestcor = cor
    return(bestloss,bestcor)

runs = [train(trainloader,net,criterion,optimizer,X_te[:,:n_feats],y_te) for i in range(3)]

print('Finished Training')
myout = net(torch.Tensor(X_te))
print('iXnos vs trained:')
print(stats.pearsonr(myout.detach().numpy().flatten(),y_te_hat.flatten()))
print('mine vs actual:')
print(stats.pearsonr(myout.detach().numpy().flatten(),ten(y_te).flatten()))
print('iXnos vs actual:')
print(stats.pearsonr(y_te_hat.flatten(),y_te.flatten()))


sumpsites = pd.read_csv('ribotranstest.all.psites.csv')
cdsdims = pd.read_csv('ribotranstest.cdsdims.csv')
cdsdims['n_cod'] = (cdsdims.stop - cdsdims.aug)/3

#okay so how much does it suffer if I leave out structure?
#a bit...


################################################################################
########Let's try it with my own generated data (hopefully teh same)
################################################################################
if True:
    trainset = Dataset(torch.Tensor(1.0*my_X_tr_s[:,:]),torch.Tensor(my_y_tr.reshape(-1,1)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                              shuffle=True, num_workers=1)
    i,data = next(enumerate(trainloader,0))
    #
    n_feats = data[0].shape[1]
    net = Net(n_feats, n_hid)
    net(data[0])
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9,nesterov=True)
    # [lrlambda(i) for i in range(0,3)]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrlambda)
    #
    train(trainloader,net,criterion,optimizer,my_X_te_s[:,:],my_y_te)

