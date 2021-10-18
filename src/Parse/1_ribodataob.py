

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotext as plx
import numpy as np
import random
import pandas as pd
import numpy as np
import glob
import h5py
import copy
import argparse
from collections import OrderedDict
import torch.utils.data as data

if not 'txtplot' in globals().keys():
    import os
    exec(open("src/0_0_rpy2plots.py").read())


ten = torch.Tensor
n_toks=512

if __name__ == '__main__':

    sys.argv=['']
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input_psites_df",default = 'ribotranstest'+'.cdsdims.csv')
    # parser.add_argument("-c", help="Fasta file", default="pipeline/yeast_transcript.ext.fa")
    parser.add_argument("-o", help="ribotrans_test", default="ribotranstest")
    args = parser.parse_args()

RIBODENS_THRESH = 0.5

if  globals().get('psitecovtbl',None) is None:
    # psitecovtbl = pd.read_csv('ribotranstest'+'.sumpsites.csv')
    psitecovtbl = pd.read_csv('../eif4f_pipeline/pipeline/ribotransdata/negIAA/negIAA.all.psites.csv')
    assert psitecovtbl.shape[0] > 1000
    cdsdims = pd.read_csv('../eif4f_pipeline/pipeline/ribotransdata/negIAA/negIAA.cdsdims.csv')
    NTOKS=512
    sumpsites = psitecovtbl
    cdsdims=cdsdims.set_index('tr_id')
    psitecovtbl = trim_middle(psitecovtbl,cdsdims,5)

class RiboTransData(data.Dataset):
    CODON_TYPES = ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG', 'ATT', 'ATC', 'ATA', 'ATG', 'GTT', 'GTC', 'GTA',
                   'GTG', 'TCT', 'TCC', 'TCA', 'TCG', 'CCT', 'CCC', 'CCA', 'CCG', 'ACT', 'ACC', 'ACA', 'ACG', 'GCT', 'GCC',
                   'GCA', 'GCG', 'TAT', 'TAC', 'CAT', 'CAC', 'CAA', 'CAG', 'AAT', 'AAC', 'AAA', 'AAG', 'GAT', 'GAC', 'GAA',
                   'GAG', 'TGT', 'TGC', 'TGG', 'CGT', 'CGC', 'CGA', 'CGG', 'AGT', 'AGC', 'AGA', 'AGG', 'GGT', 'GGC', 'GGA',
                   'GGG', 'TAA', 'TAG', 'TGA']
    genetic_code = {'TTT': 'F', 'TCT': 'S', 'TAT': 'Y', 'TGT': 'C', 'TTC': 'F', 'TCC': 'S', 'TAC': 'Y', 'TGC': 'C',
                    'TTA': 'L', 'TCA': 'S', 'TAA': '*', 'TGA': '*', 'TTG': 'L', 'TCG': 'S', 'TAG': '*', 'TGG': 'W',
                    'CTT': 'L', 'CCT': 'P', 'CAT': 'H', 'CGT': 'R', 'CTC': 'L', 'CCC': 'P', 'CAC': 'H', 'CGC': 'R',
                    'CTA': 'L', 'CCA': 'P', 'CAA': 'Q', 'CGA': 'R', 'CTG': 'L', 'CCG': 'P', 'CAG': 'Q', 'CGG': 'R',
                    'ATT': 'I', 'ACT': 'T', 'AAT': 'N', 'AGT': 'S', 'ATC': 'I', 'ACC': 'T', 'AAC': 'N', 'AGC': 'S',
                    'ATA': 'I', 'ACA': 'T', 'AAA': 'K', 'AGA': 'R', 'ATG': 'M', 'ACG': 'T', 'AAG': 'K', 'AGG': 'R',
                    'GTT': 'V', 'GCT': 'A', 'GAT': 'D', 'GGT': 'G', 'GTC': 'V', 'GCC': 'A', 'GAC': 'D', 'GGC': 'G',
                    'GTA': 'V', 'GCA': 'A', 'GAA': 'E', 'GGA': 'G', 'GTG': 'V', 'GCG': 'A', 'GAG': 'E', 'GGG': 'G'}
    codonnum = pd.Series(range(0, len(CODON_TYPES)), index=CODON_TYPES)+1
    basenum = pd.Series(range(0, 4), index=['A','C','G','T'])+1

    def __init__(self,psitecovtbl,NTOKS,RIBODENS_THRESH):
        #
        assert psitecovtbl.codon.isin(self.codonnum.index).all()

        #now filter by ribosome density (say by 0.5 - gets us top 80% of weinberg)
        ribodens = psitecovtbl.groupby('tr_id').ribosome_count.mean()
        lowdensgenes = ribodens.index[ribodens<RIBODENS_THRESH].values
        psitecovtbl = psitecovtbl[~psitecovtbl.tr_id.isin(lowdensgenes)]
        ribodens = ribodens[~ribodens.index.isin(lowdensgenes)]
        #
        # self.lowdensgenes = lowdensgenes
        self.ugenes = psitecovtbl.tr_id.unique()

        n_genes = self.ugenes.shape[0]
        print('using '+str(n_genes)+' genes')
        
        #index of the gene position
        self.gene2num = pd.Series(range(0, len(self.ugenes)), index=self.ugenes)
        #index of the codon position
        codindex = psitecovtbl.groupby('tr_id').codon_idx.rank()-1
        #ribosignal
        self.ribosignal = torch.sparse.FloatTensor(
            torch.LongTensor([
                self.gene2num[psitecovtbl.tr_id].values,
                codindex
            ]),
            torch.FloatTensor(psitecovtbl.ribosome_count.values),
            torch.Size([n_genes, NTOKS])).to_dense()
        assert self.ribosignal.shape == torch.Size([n_genes,NTOKS])

        #codons
        self.data = OrderedDict()
        self.data['codons'] = torch.sparse.FloatTensor(
            torch.LongTensor([
                self.gene2num[psitecovtbl.tr_id].values,
                codindex
            ]),
            torch.LongTensor(self.codonnum[psitecovtbl.codon].values),
            torch.Size([n_genes, NTOKS])).to_dense()

        self.data['codons'] = nn.functional.one_hot(self.data['codons']).transpose(2,1).float()
        assert self.data['codons'].shape == torch.Size([n_genes,self.codonnum.shape[0]+1,NTOKS])
        # TPMs = read_tbl[['gene', 'TPM']].drop_duplicates().set_index('gene')
        # TPMs = TPMs.TPM[ugenes]
        # TPMs = torch.FloatTensor(TPMs)
        # assert torch.isfinite(ribodens).all()
        cdslens = (self.data['codons'][:,0,:]!=1).sum(1)
        frac_filled = (cdslens/512.0)

        self.n_toks=NTOKS
        self.offset = (self.ribosignal.mean(axis=1) /  frac_filled).reshape([len(self.ugenes),1])
        self.offset = self.offset.reshape([-1,1,1])

        self.data['seqfeats'] = []
        for i in range(1,4):
            numseq = self.basenum[psitecovtbl.codon.str.split('',expand=True)[i].values].values
            denseseqfeats = torch.sparse.FloatTensor(
                torch.LongTensor([
                    self.gene2num[psitecovtbl.tr_id].values,
                    codindex
                ]),
                torch.LongTensor(numseq),
                torch.Size([n_genes, NTOKS])).to_dense()
            self.data['seqfeats'].append(nn.functional.one_hot(denseseqfeats))
        self.data['seqfeats'] = torch.cat(self.data['seqfeats'],axis=2).transpose(2,1).float()

        # non_null_mask = riboob.data['codons'][:,0,:]!=1
        # self.offset = psitecovtbl.groupby('tr_id')['ribosome_count'].mean()[self.ugenes]

        self.device = 'cpu'
        self.usedata = list(self.data.keys())
        self.batch_size=2

        self.gene2num
        self.ugenes
        self.ribosignal

        lpads = psitecovtbl.groupby('tr_id').codon_idx.min().value_counts()
        lpad = list(lpads.index)
        assert len(lpad)==1
        lpad = lpad[0]
        self.lpad = lpad
        self.batchshuffle = True

    def subset(self,idx):
        # idx=idx[idx.index[0]]
        srdata = copy.deepcopy(self)
        srdata.gene2num = srdata.gene2num[idx]
        for nm,rtensor in srdata.data.items():
            srdata.data[nm] = rtensor[srdata.gene2num.values]
        srdata.ribosignal = srdata.ribosignal[srdata.gene2num.values]
        srdata.offset = srdata.offset[srdata.gene2num.values]
        srdata.gene2num = pd.Series(range(0, len(srdata.gene2num)),index =srdata.gene2num.index)
        return srdata

       # esmgenes,esmtensor,tensname = esmtraintrs,esmtensor,'esm' 
    
    def add_tensor(self,esmgenes,esmtensor,tensname):
        nameov = pd.Series(esmgenes).isin(self.gene2num.index)
        intgenes = esmgenes[nameov]
        esmtensor = esmtensor[nameov]
        subset = self.subset(intgenes)
        subset.data[tensname]=esmtensor


    #batchinds=torch.randperm(len(self))[0:10]
    def get_batch(self,batchinds):
        device = self.device
        # genes = self.ugenes[batchinds]
        #concatenate various codon level values
        bdata = [d[batchinds] for k,d in self.data.items() if k in self.usedata]
        # bdata = [d[batchinds] for d in self.data.values()]
        bdata = torch.cat(bdata,axis=1)
        target = self.ribosignal[batchinds]
        offset = self.offset[batchinds]
        # gns = self.gene2num.index[batchinds]
        return (bdata.to(device),offset.to(device)), target.to(device)

    def datadim(self):
        return next(iter(self))[0][0].shape[1]

    def __iter__ (self):
        if(self.batchshuffle): 
            indices = torch.randperm(len(self))
        else:
            indices = list(range(len(self)))

        for batchind in range(0, len(self), self.batch_size):
            seq_len = min(self.batch_size, len(self)  - batchind)
            batchinds = indices[batchind:batchind+seq_len]
            batch = self.get_batch(batchinds)
            yield batch

    def __len__(self):
        return len(self.gene2num)

    def orfclip(self, bdata, lclip = 0, rclip = 0):
        """
        Args:
            bdata: Tensor, shape [batch, feats, pos]
            lclip: int
            rclip: int
        """
        #this function outputs a logical tensor that designates
        #positions so many places outside of ORF. lclip -1 gives you
        #1 codn upstream of start. rclip -1 clips 1 position off the
        #end of the ORF (currently no right padding on ORFs, so neg only)
        assert rclip <= 0
        assert lclip >= self.lpad
        lclip -= self.lpad
        ltens = bdata[:,0,:]!=1
        #use pandas to get max pos per gn 
        inddf = pd.DataFrame(ltens.nonzero().numpy())
        inddf.columns=['gn','pos']
        maxpos = inddf.groupby('gn').pos.max().values
        #now assign 0 or 1 to the tensor
        for g in range(bdata.shape[0]):
            ltens[g]=0
            rind = (1+maxpos[g]+rclip)
            if lclip <= rind:
                ltens[g,lclip:rind]=1
        return ltens

#this is for dev - updating our object
if not globals().get('rdata',None) is None:
    for k,v in RiboTransData.__dict__.items():
        if callable(v): 
            print(k,v)
            bound_method = v.__get__(rdata, rdata.__class__)
            rdata.__setattr__( k, bound_method)
else:
    rdata = RiboTransData(psitecovtbl,NTOKS=512,RIBODENS_THRESH=1)

if True:
    rdata.usedata=['codons','seqfeats']
    tdata = next(iter(rdata))
    assert rdata.datadim() is 80

    assert type(rdata.offset) is torch.Tensor
    assert list(tdata[0][0].shape) == [rdata.batch_size,rdata.datadim(),rdata.n_toks] 
    assert list(tdata[0][1].shape) == [rdata.batch_size,1,1] 

    assert not (next(iter(rdata))[1][0]==next(iter(rdata))[1][0]).all()
    rdata.usedata=['codons']

    assert tdata[1].shape==rdata.orfclip(tdata[0][0]).shape

if True:
    idx=rdata.gene2num.index[1:4]
    srdata = rdata.subset(idx)

    srdata.ribosignal.shape
    srdata.batch_size=3
    tdata = next(iter(srdata))
    assert list(tdata[0][0].shape) == [3,rdata.datadim(),rdata.n_toks] 
    assert list(tdata[0][1].shape) == [3,1,1]
    rdata.batchshuffle=False
    assert (next(iter(rdata))[0][0]==next(iter(rdata))[0][0]).all()
    rdata.batchshuffle=True
    assert not (next(iter(rdata))[0][0]==next(iter(rdata))[0][0]).all()


def split_data(rdata,valsize=500,tsize=500):
    n_genes = len(rdata)

    allinds = np.array(random.sample(range(0,n_genes),k=n_genes))
    tvsize=valsize+tsize
    traininds = rdata.gene2num.index[allinds[0:(len(allinds)-tvsize)]]
    testinds = rdata.gene2num.index[allinds[(len(allinds)-tvsize):(len(allinds)-valsize)]]
    valinds = rdata.gene2num.index[allinds[(len(allinds)-valsize):]]

    train_data = rdata.subset(traininds)
    test_data = rdata.subset(testinds)
    val_data = rdata.subset(valinds)
    return train_data,test_data,val_data

def split_data_vonly(rdata,valsize=1000):
    n_genes = len(rdata)

    allinds = np.array(random.sample(range(0,n_genes),k=n_genes))
    tvsize=valsize+tsize
    traininds = rdata.gene2num.index[allinds[0:(len(allinds)-tvsize)]]
    testinds = rdata.gene2num.index[allinds[(len(allinds)-tvsize):(len(allinds)-valsize)]]
    valinds = rdata.gene2num.index[allinds[(len(allinds)-valsize):]]

    train_data = rdata.subset(traininds)
    test_data = rdata.subset(testinds)
    val_data = rdata.subset(valinds)
    return train_data,test_data,val_data

train_data,test_data,val_data = split_data(rdata)

riboob=train_data

def get_codstrengths(riboob):
    ribosignal = riboob.ribosignal
    codons = riboob.data['codons']
    n_cods = codons.shape[1]
    ribodens = riboob.offset.squeeze(2)
    codstrengths = [ ((ribosignal)/(ribodens) )[riboob.data['codons'][:,i,:]==1].mean().item() for i in range(0,n_cods)]
    codstrengths = torch.FloatTensor(codstrengths[1:])
    return codstrengths

codstrengths = get_codstrengths(rdata)



if False:

    #consistent codon strengths from train to val data
    txtplot(
            np.array(get_codstrengths(train_data)),
            np.array(get_codstrengths(val_data))
            )


    #consistent ribosome densities with pandas method
    psitecovtblused = psitecovtbl[psitecovtbl.tr_id.isin(rdata.ugenes)]

    #gene offsets look like they should
    plx.clear_plot()
    plx.scatter(
        np.array(psitecovtblused.groupby('tr_id')['ribosome_count'].mean()[rdata.ugenes].values).squeeze(),
        np.array(rdata.offset).squeeze(),
        np.array(rdata.offset).squeeze(),
            rows = 17, cols = 70)
    plx.show()

    #codon level stats are same as using pandas

    traintrsums=psitecovtblused.groupby('tr_id')['ribosome_count'].sum().reset_index()

    traintrsums=traintrsums.rename({'ribosome_count':'tr_count'},axis=1)
    psitecovtblused=psitecovtblused.merge(trsums)
    psitecovtblused['dens']=psitecovtblused['ribosome_count']/psitecovtblused['tr_count']
    #
    rblcstrengths = psitecovtblused.groupby('codon')['dens'].mean()
    rblcstrengths = rblcstrengths.reset_index()
    rblcstrengths['num'] = rdata.codonnum[rblcstrengths['codon']].values
    rblcstrengths = rblcstrengths.sort_values('num')
    rblcstrengths = rblcstrengths.dens.values
    #
    plx.clear_plot()
    plx.scatter(
            np.array(get_codstrengths(train_data)),
            rblcstrengths,
            rows = 17, cols = 70)
    plx.show()


if True:
    #we can also just fake rdata
    # rdata = oldrdata
    rdata_orig = copy.deepcopy(rdata)
    fakerdata = copy.deepcopy(rdata_orig)
    fcodstrengths = torch.cat([ten([0]),codstrengths])
    fsignal = (fcodstrengths.reshape([1,65,1])*fakerdata.data['codons']).sum(axis=1)
    fsignal = fsignal*fakerdata.offset.reshape([-1,1])

    # poseffects = fakerdata.ribosignal.mean(axis=0).reshape([1,-1])
    # fsignal = fsignal*poseffects

    fakerdata.ribosignal=fsignal

    cdslens = (fakerdata.data['codons'][:,0,:]!=1).sum(1)
    frac_filled = (cdslens/512.0)

    fakerdata.n_toks=n_toks
    fakerdata.offset = (fakerdata.ribosignal.mean(axis=1) /  frac_filled).reshape([len(fakerdata.ugenes),1])
    fakerdata.offset = fakerdata.offset.reshape([-1,1,1])


    fakecodstrengths = get_codstrengths(fakerdata)
    #indeed this works.
    txtplot(fakecodstrengths,codstrengths)
    assert fakerdata.ribosignal.shape == torch.Size([rdata.ribosignal.shape[0],n_toks])
    #yup, this also works
    txtplot(fakerdata.ribosignal.sum(axis=0))

    fakerdata.offset==rdata.offset


if False:
    s_codstrengths = pd.Series(codstrengths,index=pd.Series(rdata.codonnum.index,name='codon'),name='mydens')
    s_codstrengths = pd.DataFrame({'codon':rdata.codonnum.index,'mydens':codstrengths})
    wbergsupp=pd.read_csv('../cortexomics/ext_data/weinberg_etal_2016_S2.tsv',sep='\t')
    wbergsupp = wbergsupp[['Codon','RiboDensity at A-site']]
    wbergsupp.columns = ['codon','wberg_dens']
    dtcompdf = (rdata.codonnum.
        reset_index().
        rename(columns={'index':'codon',0:'num'}).
        merge(wbergsupp).merge(s_codstrengths)
    )
    txtplot(np.log(dtcompdf.wberg_dens),np.log(dtcompdf.mydens))


rdata.batch_size=20
rdata.usedata = ['codons']
train_data,test_data,val_data = split_data(rdata,500,1)
# train_data,test_data,val_data = split_data(fakerdata,500,1)

assert train_data.batchshuffle

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(fillvalue=fillvalue, *args)

use_ixnos=True
if use_ixnos:
    from  src.IxnosTorch.ixnosdata import Ixnosdata
    # import src.IxnosTorch.train_ixmodel
    train_ixmodel = {}
    exec(
        open('src/IxnosTorch/train_ixmodel.py').read(),
        train_ixmodel)
    train_on_ixdata = train_ixmodel['train_on_ixdata']
    #cdsdims = pd.read_csv('data/ribotranstest.cdsdims.csv')
    #dataset for training
    ixdataset = Ixnosdata(psitecovtbl,cdsdims)
    trainres = train_on_ixdata(ixdataset, epochs=55)
    ixmodel = train_ixmodel['Net'].from_state_dict(vars(trainres)['beststate'])
    # Now - add ixnos as a feature in the rdata object
    # data set for 
    n_genes = rdata.data['codons'].shape[0]
    n_pos = rdata.data['codons'].shape[2]
    rdata.data['ixnos'] = torch.zeros(int(n_genes),int(n_pos))
    alltrs = list(rdata.gene2num.index)
    chunk_size = 1000
    trchunks = [alltrs[i:i + chunk_size] for i in range(0, len(alltrs), chunk_size)]
    for trchunk in trchunks:
        print('.')
        cpsitecovtbl = psitecovtbl[psitecovtbl.tr_id.isin(trchunk)]
        n_trs = len(cpsitecovtbl.tr_id.unique())
        ch_cdsdims = cdsdims[cdsdims.tr_id.isin(trchunk)]
        allposixdata = Ixnosdata(cpsitecovtbl,ch_cdsdims,
            fptrim=0,tptrim=0,tr_frac=1,top_n_thresh=n_trs)
        with torch.no_grad():
            ixpreds = ixmodel(allposixdata.X_tr.float()).detach()
        # get the cumulative bounds of these from the sizes
        cbounds = np.cumsum([0]+allposixdata.bounds_tr)
        # using the bounds, chunk our predictions
        chunkpredlist = [ixpreds[l:r] for l, r in zip(cbounds, cbounds[1:])]
        #
        assert pd.Series(allposixdata.traintrs).isin(rdata.gene2num.index).all()
        # assert pd.Series(rdata.gene2num.index).isin(psitecovtbl.tr_id).all()
        # assert pd.Series(rdata.gene2num.index).isin(allposixdata.traintrs).all()
        for i in range(len(allposixdata.traintrs)):
            itr = allposixdata.traintrs[i]
            trind = rdata.gene2num[itr]
            rdata.data['ixnos'][trind,5:(5+len(chunkpredlist[i]))]=chunkpredlist[i].reshape([-1])
    rdata.data['ixnos'] = rdata.data['ixnos'].detach()
    rdata.data['ixnos'] = rdata.data['ixnos'].reshape([-1,1,512])


    if False:#testing..
        scipy.stats.pearsonr(rdata.data['ixnos'][trind+2,5:(5+len(chunkpredlist[i]))],
            rdata.ribosignal[trind+2,6:(6+len(chunkpredlist[i]))])

        tcount = psitecovtbl.query('tr_id == @itr').query('codon_idx>=0').head(482).ribosome_count

        i=2
        itr = allposixdata.traintrs[i]
        n = len(chunkpredlist[i])
        tcount = psitecovtbl.query('tr_id == @itr').query('codon_idx>=0').head(n).ribosome_count
        txtplot(chunkpredlist[i].flatten(),tcount)
        tcount = tcount - tcount.mean()
        tcount = tcount / tcount.std()
        txtplot(chunkpredlist[i].flatten(),tcount)
        # [len(c) for c in chunkpredlist]
        assert rdata.data['ixnos'].sum(axis=1).min()>0

if True:
    #Load good elongation rates, does my ixos here reflect them?
    #Can I then get those out the other end of my transformer?
    goodelong = pd.read_csv('../eif4f_pipeline/pipeline/ixnos_elong/negIAA/negIAA.elong.csv')
    trchunk = ixdataset.traintrs
    cpsitecovtbl = psitecovtbl[psitecovtbl.tr_id.isin(trchunk)]
    n_trs = len(cpsitecovtbl.tr_id.unique())
    ch_cdsdims = cdsdims[cdsdims.tr_id.isin(trchunk)]
    allposixdata = Ixnosdata(cpsitecovtbl,ch_cdsdims,
        fptrim=0,tptrim=0,tr_frac=1,top_n_thresh=n_trs)
    allposixdata.y_tr==ixdataset.y_tr
    with torch.no_grad():
        ixpreds = ixmodel(allposixdata.X_tr.float()).detach()
    # get the cumulative bounds of these from the sizes
    cbounds = np.cumsum([0]+allposixdata.bounds_tr)
    # using the bounds, chunk our predictions
    chunkpredlist = [ixpreds[l:r] for l, r in zip(cbounds, cbounds[1:])]
    emeans = [x.mean().item() for x in chunkpredlist]
    myelong = pd.DataFrame([trchunk,pd.Series(emeans)]).transpose()
    myelong.columns = ['tr_id','myelong']
    myelong.myelong = pd.Series([float(x) for x in myelong.myelong.values])
    myelong=myelong.merge(goodelong)
    scipy.stats.pearsonr(myelong.myelong.values,
        myelong.elong.values)
    trmyelong.columns=['tr_id','trmyelong']
    myelong=myelong.merge(trmyelong)
    scipy.stats.pearsonr(myelong.myelong.values,
        myelong.trmyelong.values)

    scipy.stats.pearsonr(allposixdata.y_tr,
        [z.item() for x in chunkpredlist for z in x])

    scipy.stats.pearsonr(allposixdata.y_tr,
        [z.item() for x in chunkpredlist for z in x])

    with torch.no_grad():
        trixpreds = ixmodel(ixdataset.X_tr.float()).detach()

    scipy.stats.pearsonr(ixdataset.y_tr,
        trixpreds[:,0])

    with torch.no_grad(): 
        trixpreds = ixmodel(ixdataset.X_tr.float()).detach() 
    #do the preds from the training one match the good ones from the pipeline    
    cbounds = np.cumsum([0]+ixdataset.bounds_tr)
    # using the bounds, chunk our predictions
    chunkpredlist = [trixpreds[l:r] for l, r in zip(cbounds, cbounds[1:])]
    emeans = [x.mean().item() for x in chunkpredlist]
    trmyelong = pd.DataFrame([pd.Series(ixdataset.traintrs),pd.Series(emeans)]).transpose()
    trmyelong.columns = ['tr_id','trmyelong']
    trmyelong.trmyelong = pd.Series([float(x) for x in trmyelong.trmyelong.values])
    trmyelong=trmyelong.merge(goodelong)
    trmyelong=trmyelong[~np.isnan(trmyelong.trmyelong)]
    scipy.stats.pearsonr(trmyelong.trmyelong,
        trmyelong.elong)
    txtplot(trmyelong.trmyelong,
        trmyelong.elong)

    ##ooookay so at least these elongs are sane and match the good ones.

        #[z.item() for x in chunkpredlist for z in x])


#the stacking doesn't work

