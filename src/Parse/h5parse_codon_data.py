if True:
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

# var_exists = 'test_target' in locals() or 'test_target' in globals()

# if not var_exists:

GET_ESM_TOKENS = False

# +
if not 'USE_SEQ_FEATURES' in globals().keys():
    USE_SEQ_FEATURES = True
if not 'USE_ESM_TOKENS' in globals().keys():
    GET_ESM_TOKENS = True

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

# -

# ribodatafile = '../Liuetal_pipeline/pipeline/ribotrans_process/ribo_0h/ribotrans.csv.gz'
# ribodatafile = '../ribotrans_process/ribo_0h/ribotrans.csv.gz'
ribodatafile = 'yeast_test.csv'
# esmweightfolder = '/scratch/AG_Ohler/dharnet/Gencodev24lift37_tokens/'

# esmweightfolder = '/fast/scratch/users/dharnet_m/tmp/Gencodev24lift37_tokens/'
esmweightfolder = '/fast/scratch/users/dharnet_m/tmp/yeast_tokens/'
esmfiles = glob.glob(esmweightfolder+'/*')

AMINO_ACIDS = ['A', 'R', 'D', 'N', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']
if not 'read_tbl' in globals():
 print('reading reads file')
 read_tbl = pd.read_csv(
    ribodatafile,
    usecols = ['codon_idx','gene','ribosome_count','TPM','codon']
    )
print('using reads file '+ ribodatafile)

# ## bash code etc for fetching data
#

# +
# fa=../Liuetal_pipeline/ext_data/gencode.v24lift37.pc_translations.fa ;
# fa=../Liuetal_pipeline/ext_data/gencode.v24lift37.pc_translations.fa ;
# fa='yeast_transl.fa'
# localfa=$(basename ${fa%.fa}.trid.fa )
# #cat $fa | perl -lanpe 's/>.*\|(ENST\w+\.\d+).*/>\1/' > $localfa
# head $localfa
# python Applications/esm/extract.py esm1_t34_670M_UR50S $localfa Gencodev24lift37_tokens  --include per_tok && echo 'finished!'
# python Applications/esm/extract.py esm1_t34_670M_UR50S $localfa Gencodev24lift37_tokens  --include per_tok && echo 'finished!'
# python Applications/esm/extract.py esm1_t34_670M_UR50S yeast_transl.fa  ~/scratch/tmp/yeast_tokens    --include per_tok && echo 'finished!'
# -


#parse transcript names out of the esmfile names
esmweightfolder = '/fast/scratch/users/dharnet_m/tmp/yeast_tokens/'
esmfiles = glob.glob(esmweightfolder+'/*')
if 'yeast' in ribodatafile:
    esmtrs =pd.Series(esmfiles).str.extract('.*/(.*?).pt')[0].values    
else:
    esmtrs =pd.Series(esmfiles).str.extract('(ENST\\d+)')[0].values
esmfiles = pd.Series(esmfiles,index=esmtrs)


n_cods = len(CODON_TYPES)+1 #There needs to be an uknown 
n_len = 512
n_toks = torch.load(esmfiles[1])['representations'][34].shape[1]

#now i can subset the rdata tensor with the coon tensor.

if True:
    print('parsing to tensors...')
 
    reads2use = (read_tbl.
                 query('codon_idx >= -10').
                 query('codon_idx < (512 -10 )')
                 # query('ribosome_count!=0').
                 # query('TPM>8')
                 )

    reads2use = reads2use[reads2use.codon.isin(codonnum.index)]

    ribodens = reads2use.groupby('gene').ribosome_count.mean()
    lowdensgenes = ribodens.index[ribodens<5].values
    reads2use = reads2use[~reads2use.gene.isin(lowdensgenes)]
    ribodens = ribodens[~ribodens.index.isin(lowdensgenes)]

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
    n_genes = ugenes.shape[0]                    

    print('using '+str(n_genes)+' genes')

    print('try new tpm method')
    ribodens = ribodens[ugenes].values
    ribodens = ribodens**-1
    ribodens = torch.FloatTensor(ribodens).log()


    gene2num = pd.Series(range(0, len(ugenes)), index=ugenes)
    
    assert reads2use.codon.isin(codonnum.index).all()
    assert torch.isfinite(ribodens).all()

    poseffects = True

    ribosignal = torch.sparse.FloatTensor(
        torch.LongTensor([
            gene2num[reads2use.gene].values,
            reads2use.codon_idx.values+10
        ]),
        torch.FloatTensor(reads2use.ribosome_count.values),
        torch.Size([n_genes, n_len])).to_dense()
    
    # TPMs = torch.log(ribosignal.mean(axis=1)**-1)

    codons = torch.sparse.FloatTensor(
        torch.LongTensor([
            gene2num[reads2use.gene].values,
            reads2use.codon_idx.values+10
        ]),
        torch.LongTensor(codonnum[reads2use.codon].values),
        torch.Size([n_genes, n_len])).to_dense()

    codons = nn.functional.one_hot(codons).transpose(2,1).float()

    assert codons.shape == torch.Size([n_genes,n_cods,n_len])
    assert ribosignal.shape == torch.Size([n_genes,n_len])

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
                    gene2num[reads2use.gene].values,
                    reads2use.codon_idx.values+10
                ]),
                torch.LongTensor(numseq),
                torch.Size([n_genes, n_len])).to_dense()
            seqfeats.append(nn.functional.one_hot(denseseqfeats))

        seqfeats = torch.cat(seqfeats,axis=2).transpose(2,1).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bptt=100
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

esmtensororig = esmtensor

h5file = 'testfile.h5'

testfile = h5py.File(h5file, 'w')
testfile['data'] = torch.eye(1000)


h5ob = h5py.File(h5file, 'r')
torch.from_numpy(np.array(h5ob['data'][0:3,0:3]))

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
    




