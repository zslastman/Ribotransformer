if True:
	import math
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import plotext as plx
	import numpy as np
	import random


# var_exists = 'test_target' in locals() or 'test_target' in globals()

# if not var_exists:

poseffects = True

if True:
	n_cods=64
	# n_cods=5
	n_genes = 6000
	# n_genes = 3
	n_len = 512
	#
	fakecodonseqs = torch.LongTensor(n_len,n_genes).random_(0,n_cods)
	fakecodonseqs1h = nn.functional.one_hot(fakecodonseqs)
	#
	codstrengths = 1.05**np.linspace(-1-(60/2),2+(60/2),4+60)
	codstrengths = torch.FloatTensor(codstrengths)
	#
	codsig = fakecodonseqs1h.float().matmul(codstrengths)

	starteffect = np.concatenate([1.5**(-np.linspace(0,10,10)),np.array([0]*(512-10))])
	stopeffect = np.concatenate([np.array([0]*(512-10)),1.5**(-np.linspace(0,10,10))])

	genetpmvect = 2*(10**np.linspace(1,4,n_genes))/n_len
	genetpmvect = genetpmvect/genetpmvect
	genetpmvect = torch.FloatTensor(genetpmvect)

	#n,n_genes codsig by (n_genes,n_genes) tpmmatrix gets us a n,n_genes signal mat
	fakesignal = codsig * genetpmvect.reshape(1,n_genes)
	if poseffects:
		fakesignal = fakesignal * np.exp(starteffect.reshape([n_len,1])) * np.exp(stopeffect.reshape([n_len,1]))


	fakesignal = torch.poisson(fakesignal)
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
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	TPMoffsets = torch.log(fakesignal.mean(axis=1)**-1)
	
    # source, targetsource, offsetsource, i = train_data, train_regdata, train_offsets, i

	def cget_batch(source, targetsource, offsetsource, i, bptt=100,device=device):
	    seq_len = min(bptt, len(source) - 1 - i)
	    data = source[i:i+seq_len]
	    target = targetsource[i:i+seq_len].unsqueeze(1)
	    offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
	    return data.to(device), target.to(device), offset.to(device)

	assert list(cget_batch(fakecodonseqs1h, fakesignal, TPMoffsets, 1)[0].shape)==[100,n_cods,n_len]
	assert list(cget_batch(fakecodonseqs1h, fakesignal, TPMoffsets, 1)[1].shape)==[100,1,n_len]
	assert list(cget_batch(fakecodonseqs1h, fakesignal, TPMoffsets, 1)[2].shape)==[100,1,1]

	allinds = np.array(random.sample(range(0,n_genes),k=n_genes))

	traininds = allinds[0:(len(allinds)-1000)]
	testinds = allinds[(len(allinds)-1000):(len(allinds)-500)]
	valinds = allinds[(len(allinds)-500):]

	train_data, train_target = fakecodonseqs1h[traininds],fakesignal[traininds]
	val_data, val_target = fakecodonseqs1h[valinds],fakesignal[valinds]
	test_data, test_target = fakecodonseqs1h[testinds],fakesignal[testinds]
	
	#we add these to the output of our model, to normalize for TPM	
	train_offsets = TPMoffsets[traininds]
	val_offsets = TPMoffsets[valinds]
	test_offsets = TPMoffsets[testinds]


    #we add these to the output of our model, to normalize for TPM  
(fakesignal * fakecodonseqs1h[:,1,:]).shape
(TPMoffsets.unsqueeze(1)).exp().shape


#calculate average signal per codon - just proving this works for when i do it with data
pcodstrengths = [ ((fakesignal)/(TPMoffsets.exp().unsqueeze(1)))[fakecodonseqs1h[:,i,:]==1].mean().item() for i in range(0,n_cods)]
pcodstrengths = torch.FloatTensor(pcodstrengths)
plx.clear_plot()
plx.scatter(
        codstrengths.cpu().numpy().flatten(),
        pcodstrengths.cpu().numpy().flatten(),
        rows = 17, cols = 70)
plx.show()	
fakesignal.sum(axis=1).max()
fakesignal.sum(axis=1).min()

# traininds.shape
# testinds.shape
# valinds.shape
# train_data.shape
# train_target.shape
# val_data.shape
# val_target.shape
# test_data.shape
# test_target.shape
# train_offsets.shape
# val_offsets.shape
# test_offsets.shape