import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotext as plx
import numpy as np
import random
import ipdb
import time
import torch
import numpy as np

bn = 1
c_in = 2
L = 5
cout=1

k=5
kc = int(k/2)

track = torch.Tensor(np.zeros([bn,c_in,L]))


track[0,0,2] = 5
track[0,1,2] = 3

cutability = torch.Tensor([5,.004,3,2,1]).reshape([1,1,5])
mcut = torch.Tensor(np.zeros([kc+1,1,k]))
#note setting it up this way stacks information from further BEHIND 
#each bp (value of L) on top of the value for that bp.
# mcut[:,0,:] =  torch.diag(torch.Tensor([1,100,10,0,0]))
dvect = [1]*3
dvect[2]=100
dvect[0]=50
i=0
for i in range(kc+1): mcut[i,0,kc-i] = dvect[i]

# This mcut vector COULD be multiplied by the 5' distance probabilites
# which would, over each position, stack up the cutability multiplied by the
# distance factor
# which could then be normalized.
# you could also cross multiply it by the 3' cut probability.
# then we could shift it back, so that each position has a vector of
# how likely each posiiton AHEAD of it is to give it a read, of a given length
# Then for each position and length you'd pick the maximum.

# mcut[0,0,0] = 1
# mcut[1,0,1] = 1
# mcut[2,0,2] = 1
# mcut[3,0,3] = 1
# mcut[2,0,2] = 1
cutability
fpcutprobs = F.conv1d(cutability,mcut,padding=int(k/2))
fpcutprobs
#b,kc,L vector where  
fpcutprobs = fpcutprobs / fpcutprobs.sum(axis=[0,1])

#
#now I need to figure out a convolution that undoes the stacking
m2cut = torch.Tensor(np.zeros([kc+1,kc+1,k]))
# m2cut[:,0,] =  torch.diag(torch.Tensor([1,1,1,0,0]))
# m2cut[0,0,2]=1
# m2cut[1,1,3]=1
# m2cut[kc,kc,kc]=1#for channel kc, fetch 1 of channel kc from kc ahead
# m2cut[kc+1,kc-1,kc+1]=1#for channel kc, fetch 1 of channel kc from kc ahead
# m2cut[kc+2,kc-2,kc+2]=1#for channel kc, fetch 1 of channel kc from kc ahead
# m2cut[1,1,4]=1
# m2cut[1,1,3]=1
# m2cut[2,0,4]=1
# m2cut[4,4,4]=1
# m2cut[3,3,3]=1
for i in range(kc+1): m2cut[i,i,kc+i] = dvect[i]
fp_orig_prop = F.conv1d(fpcutprobs, m2cut, padding=int(k/2))
#so now we have a batch,k,L object
#that expresses the amount of signal from  l+k that will land on l
fp_orig_prop

fp_orig_prop = fp_orig_prop / fp_orig_prop.sum(axis=[0,1])
#We could use this to shift an L vector forward I guess..


fp_orig_prop.argmax(1)


fp_orig_prop*10
(fp_orig_prop*100).floor()


mcut[0,0,:] = mcut[0,0,:] / mcut[0,0,:].sum() 
mcut[0,0,:] = mcut[0,0,:] / mcut[0,0,:].sum() 




#
k=5
kc = int(k/2)

shift=1#
# m [outputchannel,weights_on_input_channels,input offset]
#so e.g. kc - 2 would result in input shifted forward by a certain amount
m = torch.Tensor(np.zeros([cout,c_in,k]))
# m[:,:,kc-shift] = torch.Tensor([
			 # [1,0],
			 # ])
# m[0,:,kc:kc-shift]
m[0,0,:] = torch.Tensor([0,0,3,5,2])
m[0,1,:] = torch.Tensor([0,0,3,5,2])
m[0,0,:] = m[0,0,:] / m[0,0,:].sum() 
m[0,1,:] = m[0,1,:] / m[0,1,:].sum() 

m[0,0,:] = torch.Tensor([0,0,3,5,2])
m[0,1,:] = torch.Tensor([0,0,3,5,2])
m[0,0,:] = m[0,0,:] / m[0,0,:].sum() 
m[0,1,:] = m[0,1,:] / m[0,1,:].sum() 


#
out = F.conv1d(track,weight=m,padding=int(k/2))




track
out
out.sum()
track.sum()

out.shape
out[5,:,:]
track[5,:]
#so if I start with an occupancy vect, and transform it back, I face teh issue of
#actually applying this to the data.
#strategy could be, N O vect,  N x l Of vect (and later Ot), then transform Of using conv with surrounding sequence
# or start with cutability vect, for each location 


#