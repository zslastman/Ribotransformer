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
if 'yeast' in ribodatafile:
    esmtrs =pd.Series(esmfiles).str.extract('.*/(.*?).pt')[0].values    
else:
    esmtrs =pd.Series(esmfiles).str.extract('(ENST\\d+)')[0].values
esmfiles = pd.Series(esmfiles,index=esmtrs)


codonnum = pd.Series(range(0, len(CODON_TYPES)), index=CODON_TYPES)+1
basenum = pd.Series(range(0, 4), index=['A','C','G','T'])+1
n_cods = len(CODON_TYPES)+1 #There needs to be an uknown 
n_len = 512
n_toks = torch.load(esmfiles[1])['representations'][34].shape[1]


fasta='/fast/work/groups/ag_ohler/dharnet_m/cortexomics/yeast_test/Yeast.saccer3.fa'
gtf = '/fast/work/groups/ag_ohler/dharnet_m/cortexomics/yeast_test/Yeast.sacCer3.sgdGene.gtf'
cdsext,cdsseq,codons = load_anno(gtf, fasta, 0)


riboh5 = 'ext_data/2016_Weinberg_RPF.h5'


def firstval(dict):
    return next(iter(dict.values()))

#We maybe want to leave out the middle actually, but that can wait...
n_len=1024
rmatsize = n_len*3
tlbuff= 10*3
tlbuff=0

#def parse_riboviz_h5(riboh5):
if True:
    h5f = h5py.File(riboh5,'r')
    h5genes = list(h5f.keys())
    gene1 = h5genes[666]

    rdatalist=[]
    for gene1 in h5genes:
        d1 = h5f[gene1]
        dsetname = list(d1.keys())[0]
        d2 = d1[dsetname]
        readsname = list(d2.keys())[0]
        readsdata = h5f[gene1][dsetname][readsname]
        startpos = readsdata.attrs['start_codon_pos'][0]
        stoppos = readsdata.attrs['stop_codon_pos'][0]
        readsdata['data'][startpos-1:stoppos]
        #shows names of attributes

        trlen = readsdata['data'].shape[0]
        lbuff = int(readsdata.attrs['buffer_left'])
        rbuff = int(readsdata.attrs['buffer_right'])
        cdslen = int(trlen - lbuff - rbuff)

        if cdslen > rmatsize: continue  
        
        cdsstartp0 = (startpos-1)
        cdsstopp0 = (startpos-1+cdslen)
        cdsstopp0 = (stoppos-2)

        #for each cds well take 


        readsdata['data'][cdsstartp0:cdsstartp0+tlbuff+rmatsize+1]

        rdata = readsdata['data'][cdsstartp0:cdsstopp0]
        bufferbps = rmatsize - rdata.shape[0]
        rbuffer = np.zeros([bufferbps,rdata.shape[1]])
        rdata= np.concatenate([rdata,rbuffer])
        rdatalist.append((rdata,gene1,lbuff,cdslen))

    rlens = readsdata.attrs['lengths']


rdatatens = np.stack([i[0] for i in rdatalist])
keeplens = (rlens >= 25) &(rlens <= 35)
rlens = rlens[keeplens]

rdatatens = rdatatens[:,:,keeplens]

rdatagenes = [i[1] for i in rdatalist]
gene2num = pd.Series(range(0, len(rdatagenes)), index=rdatagenes)

codonsdf= codons[codons.gene.isin(rdatagenes)]

codons_ts = torch.sparse.FloatTensor(
        torch.LongTensor([
            gene2num[codonsdf.gene].values,
            codonsdf.codon_idx.values+tlbuff
        ]),
        torch.LongTensor(codonnum[codonsdf.codon].values),
        torch.Size([rdatatens.shape[0],int(rmatsize/3) ])).to_dense()

#note this endds up w62 codons tall cos the stops are missing
rdcodons = nn.functional.one_hot(codons_ts).transpose(2,1).float()

assert codons.shape==torch.Size([len(rdatagenes),62,n_len])
assert readsdata.shape==torch.Size([len(rdatagenes),n_rls])

rdatatens = rdatatens

phase=0
codonind=3
rlind = 0
position=1#codon position relative to codon

cinds = range(0,rdcodons.shape[1])
positions = [int(p) for p in range(-12,3) ]
phases = [0,1,2]

def codonloc(rdcodons,codonind,position):
    codonlocs = rdcodons[:,codonind,:]==1
    codonlocs = codonlocs.roll(dims=1,shifts=position)
    return codonlocs
    
#normalize 
rdatanorm =  (rdatatens/np.expand_dims(rdatatens.sum(axis=(1,2))+1,axis=[1,2]))

#get metacodons
posrl_dens = [ [rdatanorm[:,phase::3,:][codonloc(rdcodons,cind,pos)].mean(axis=0)  for pos in positions] for cind in cinds]
posrl_dens = np.stack(posrl_dens)
posrlstddev = posrl_dens.std(axis=0)
posrlstddev /= posrl_dens.mean(axis=0)

def txtplot(x,y):
    plx.clear_plot()
    plx.scatter(x,y,rows = 20, cols = 40)
    plx.show()

txtplot(positions,posrlstddev[:,6])







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
        target = targetsource[i:i+seq_len].unsqueeze(1)
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



n_feats = combdata.shape[1]
n_traingenes = combdata.shape[0]
print('Got '+str(n_feats)+' features for '+str(n_traingenes))
codstrengths.exp().sort()
