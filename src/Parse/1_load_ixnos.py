import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotext as plx
import numpy as np
import random
import ipdb
import time
import scipy
from  scipy import stats
    
modelobs = ['X_te',  'X_tr', 'X_val','y_te', 'y_te_hat', 'y_tr', 'y_tr_hat', 'y_val']
finalmodelobs = [np.loadtxt(f'ext_data/Ixnos/{modelob}.txt') for modelob in modelobs]
X_te,  X_tr, X_val,y_te, y_te_hat, y_tr, y_tr_hat, y_val  = finalmodelobs

finalmodelweights = [np.loadtxt(f'ext_data/Ixnos/modelweights_{i}.txt'.format(i=i)) for i in range(0,4)]
# import pickle;
# pickle.dump((),open('.p','wb'))
# = pickle.load(open('.p','rb'))

