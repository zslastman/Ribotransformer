
import math
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

RIBODENS_THRESH = 0.5

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

reads2use = sumpsites

assert reads2use.codon.isin(codonnum.index).all()



#now filter by ribosome density (say by 0.5 - gets us top 80% of weinberg)
ribodens = reads2use.groupby('tr_id').ribosome_count.mean()
lowdensgenes = ribodens.index[ribodens<RIBODENS_THRESH].values
reads2use = reads2use[~reads2use.tr_id.isin(lowdensgenes)]
ribodens = ribodens[~ribodens.index.isin(lowdensgenes)]

ugenes = reads2use.tr_id.unique()

print('using '+str(n_genes)+' genes')

 

n_genes = ugenes.shape[0]


gene2num = pd.Series(range(0, len(ugenes)), index=ugenes)

codindex = reads2use.groupby('tr_id').codon_idx.rank()-1

ribosignal = torch.sparse.FloatTensor(
    torch.LongTensor([
        gene2num[reads2use.tr_id].values,
        codindex
    ]),
    torch.FloatTensor(reads2use.ribosome_count.values),
    torch.Size([n_genes, NTOKS])).to_dense()

codons = torch.sparse.FloatTensor(
    torch.LongTensor([
        gene2num[reads2use.tr_id].values,
        codindex
    ]),
    torch.LongTensor(codonnum[reads2use.codon].values),
    torch.Size([n_genes, NTOKS])).to_dense()

codons = nn.functional.one_hot(codons).transpose(2,1).float()
assert codons.shape == torch.Size([n_genes,codonnum.shape[0]+1,NTOKS])
assert ribosignal.shape == torch.Size([n_genes,NTOKS])


# TPMs = read_tbl[['gene', 'TPM']].drop_duplicates().set_index('gene')
# TPMs = TPMs.TPM[ugenes]
# TPMs = torch.FloatTensor(TPMs)
# assert torch.isfinite(ribodens).all()

ribodens = torch.log(ribosignal.mean(axis=1)**-1)

if USE_SEQ_FEATURES:
    i=1        
    seqfeats = []
    for i in range(1,4):
        numseq = basenum[reads2use.codon.str.split('',expand=True)[i].values].values
        denseseqfeats = torch.sparse.FloatTensor(
            torch.LongTensor([
                gene2num[reads2use.tr_id].values,
                codindex
            ]),
            torch.LongTensor(numseq),
            torch.Size([n_genes, NTOKS])).to_dense()
        seqfeats.append(nn.functional.one_hot(denseseqfeats))

    seqfeats = torch.cat(seqfeats,axis=2).transpose(2,1).float()




import pickle;
pickle.dump((),open('.p','wb'))
 = pickle.load(open('.p','rb'))

import torch.utils.data as data

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

    def __init__(self,reads2use):
        #
        assert reads2use.codon.isin(self.codonnum.index).all()

        #now filter by ribosome density (say by 0.5 - gets us top 80% of weinberg)
        ribodens = reads2use.groupby('tr_id').ribosome_count.mean()
        lowdensgenes = ribodens.index[ribodens<RIBODENS_THRESH].values
        reads2use = reads2use[~reads2use.tr_id.isin(lowdensgenes)]
        ribodens = ribodens[~ribodens.index.isin(lowdensgenes)]
        #
        # self.lowdensgenes = lowdensgenes
        ugenes = reads2use.tr_id.unique()


        n_genes = ugenes.shape[0]
        print('using '+str(n_genes)+' genes')
        
        #index of the gene position
        self.gene2num = pd.Series(range(0, len(ugenes)), index=ugenes)
        #index of the codon position
        codindex = reads2use.groupby('tr_id').codon_idx.rank()-1
        #ribosignal
        self.ribosignal = torch.sparse.FloatTensor(
            torch.LongTensor([
                self.gene2num[reads2use.tr_id].values,
                codindex
            ]),
            torch.FloatTensor(reads2use.ribosome_count.values),
            torch.Size([n_genes, NTOKS])).to_dense()
        assert self.ribosignal.shape == torch.Size([n_genes,NTOKS])

        #codons
        self.codons = torch.sparse.FloatTensor(
            torch.LongTensor([
                self.gene2num[reads2use.tr_id].values,
                codindex
            ]),
            torch.LongTensor(self.codonnum[reads2use.codon].values),
            torch.Size([n_genes, NTOKS])).to_dense()

        self.codons = nn.functional.one_hot(self.codons).transpose(2,1).float()
        assert self.codons.shape == torch.Size([n_genes,self.codonnum.shape[0]+1,NTOKS])
        # TPMs = read_tbl[['gene', 'TPM']].drop_duplicates().set_index('gene')
        # TPMs = TPMs.TPM[ugenes]
        # TPMs = torch.FloatTensor(TPMs)
        # assert torch.isfinite(ribodens).all()
        self.ribodens = torch.log(ribosignal.mean(axis=1))

        i=1        
        self.seqfeats = []
        for i in range(1,4):
            numseq = self.basenum[reads2use.codon.str.split('',expand=True)[i].values].values
            denseseqfeats = torch.sparse.FloatTensor(
                torch.LongTensor([
                    gene2num[reads2use.tr_id].values,
                    codindex
                ]),
                torch.LongTensor(numseq),
                torch.Size([n_genes, NTOKS])).to_dense()
            self.seqfeats.append(nn.functional.one_hot(denseseqfeats))
        self.seqfeats = torch.cat(self.seqfeats,axis=2).transpose(2,1).float()

        # return(ribodens,ribosignal,codons,seqfeats)

        # self.lowdensgenes
        self.gene2num
        self.codons
        self.ribosignal
        self.seqfeats
        self.rtensornms = ['codons','ribosignal','seqfeats']


    def subset(self,idx):
        # idx=idx[idx.index[0]]
        srdata = copy.deepcopy(self)
        srdata.gene2num = srdata.gene2num[idx]
        for rtensor in srdata.rtensornms:
            subtensor = getattr(srdata,rtensor)[srdata.gene2num.values]
            srdata.__setattr__(rtensor,subtensor)
        srdata.gene2num = pd.Series(range(0, len(srdata.gene2num)),index =srdata.gene2num.index)
        return srdata
        
    def add_tensor(self,esmgenes)
        intgenes = esmgenes[pd.Series(esmgenes).isin(rdata.gene2num.index)]
        self = self.subset(intgenes)
        self.__setattr__(tensname,esmtensor)
        if not tensname in self.rtensornms: self.rtensornms += [tensname]


    def cget_batch(source, targetsource, offsetsource, i, device, bptt=100):
        if device is None:device = torch.device('cpu')
        seq_len = min(bptt, len(source) - 1 - i)
        data = self.get_data[i:i+seq_len]
        # target = targetsource[i:i+seq_len]
        target = targetsource[i:i+seq_len].unsqueeze(1)
        # offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
        offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
        return data.to(device), target.to(device), offset.to(device)

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(srdata.gene2num)
#this is for dev - updating our object
if not globals().get('rdata',None) is None:
    for k,v in RiboTransData.__dict__.items():
        if callable(v): 
            print(k,v)
            bound_method = v.__get__(rdata, rdata.__class__)
            rdata.__setattr__( k, bound_method)
        # rdata.__setattr__(k,v)
# rdata.subset(idx).codons.shape

#
rdata = RiboTransData(reads2use)
idx = gene2num.index[1:4]

srdata = rdata.subset(idx)

#now let's add the tensor data
tokentensorfile = 'yeast_test_esmtensor.pt'
esmgenes,esmtensor = torch.load(open(tokentensorfile,'rb'))

tensname='esmtokens'
esmrdata.add_tensor(esmgenes,esmtensor,tensname):
    # self=

esmrdata.get_batch()



def add_tokens(rdata,ugenes,esmtensor):


add_tokens(rdata,ugenes,esmtensor)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bptt=100



print('try new tpm method')
ribodens = ribodens[ugenes].values
ribodens = ribodens**-1
ribodens = torch.FloatTensor(ribodens).log()



    if GET_ESM_TOKENS:
        assert len(esmfiles)>0
        if not 'esmtensor' in globals().keys():
            try:
                ugenes,esmtensor = torch.load(open(ribodatafile.replace('.csv','_')+'esmtensor.pt','rb'))
                n_genes = ugenes.shape[0]

            except FileNotFoundError:

                ugenes = reads2use.gene.unique()
                ugenes.sort()

                if 'yeast' in ribodatafile:
                    ugenetrs=pd.Series(ugenes)
                else:
                    ugenetrs = pd.Series(ugenes).str.extract('(ENSTR?\\d+)')
                    foo #fix this so it's a simple series
          
                hasesm = ugenetrs.isin(esmfiles.index)
                # ugenes = ugenes[hasesm.values[:,0]]
                ugenes = ugenes[hasesm.values]
                ugenetrs = ugenetrs[hasesm.values]
                ugeneesmfiles = esmfiles[ugenetrs.values]

                n_genes = ugenes.shape[0]

                print('parsing ESM data')
                esmtensor = torch.zeros([n_genes,n_toks,n_len])
                for i,esmfile in enumerate(ugeneesmfiles):
                    if not i % 10: print('.')
                    esmdat = torch.load(esmfile)['representations'][34]
                    esmdat = esmdat[0:(512-10),:]
                    esmtensor[i,:,10:(esmdat.shape[0]+10)] = esmdat.transpose(0,1)
            
                torch.save([ugenes,esmtensor],  ribodatafile.replace('.csv','_')+'esmtensor.pt')

        ugenesinreads = pd.Series(ugenes).isin(reads2use.gene)
        ugenes = ugenes[ugenesinreads]
        esmtensor = esmtensor[ugenesinreads]
        reads2use = reads2use[reads2use.gene.isin(ugenes)]
        n_genes = ugenes.shape[0]   
        assert esmtensor.shape == torch.Size([n_genes,n_toks,n_len])

    else:    
        ugenes = reads2use.gene.unique()
        ugenes.sort()









    def cget_batch(source, targetsource, offsetsource, i, bptt=100,device=device):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        # target = targetsource[i:i+seq_len]
        target = targetsource[i:i+seq_len].unsqueeze(1)
        # offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
        offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
        return data.to(device), target.to(device), offset.to(device)

    assert [x.shape for x in cget_batch(codons, ribosignal, ribodens, 1)] ==  [
         torch.Size([bptt, n_cods, n_len ]),
         torch.Size([bptt, 1, n_len]),
         torch.Size([bptt, 1, 1])]


    print('splitting...')
    allinds = np.array(random.sample(range(0,n_genes),k=n_genes))
    traininds = allinds[0:int(n_genes*.6)]
    testinds = allinds[(int(n_genes*.6)):int(n_genes*.8)]
    valinds = allinds[int(n_genes*.8):]

    esmmax = esmtensor.max(0).values.max(1).values
    esmmin = esmtensor.min(0).values.min(1).values

    esmmax = esmmax.reshape([1,n_toks,1])
    esmmin = esmmin.reshape([1,n_toks,1])

    assert not  (esmmax == esmmin).any()

    esmtensornorm = (esmtensor - esmmin) / (esmmax - esmmin )


#calculate average signal per codon
codstrengths = [ ((ribosignal)/(ribodens.exp().unsqueeze(1)) )[codons[:,i,:]==1].mean().item() for i in range(0,n_cods)]
codstrengths = torch.FloatTensor(codstrengths[:]).log()




################################################################################
########H5 encoding of data
################################################################################
import h5py
import os
esmtensororig = esmtensor

h5file = 'testfile.h5'
testfile = h5py.File(h5file, 'w')
os.remove(h5file)


testfile['data'] = torch.eye(1000)

testfile['data'] = ribosignal

testfile['ribosignal'] = ribosignal

testfile['ribosignal'].resize(testfile['ribosignal'].shape[0]+10**4, axis=0)

h5ob = h5py.File(h5file, 'r')
torch.from_numpy(np.array(h5ob['data'][0:3,0:3]))


with h5py.File(path, "a") as f:
    dset = f.create_dataset('ribosignal', (10**5,), maxshape=(None,),
                            dtype='i8', chunks=(10**4,))
    dset[:] = np.random.random(dset.shape)        
    print(dset.shape)
    # (100000,)

    for i in range(3):
        dset.resize(dset.shape[0]+10**4, axis=0)   
        dset[-10**4:] = np.random.random(10**4)
        print(dset.shape)
        # (110000,)
        # (120000,)
        # (130000,)




class HDF5DatasetSingle(data.Dataset):
    """Represents an abstract HDF5 dataset that's in a single file.
    
    Input params:
        file_path: Path to the hdf5 file.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))
    
















