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

# var_exists = 'test_target' in locals() or 'test_target' in globals()

# if not var_exists:


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

AMINO_ACIDS = ['A', 'R', 'D', 'N', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']


if not 'read_tbl' in globals():
 print('reading reads file')
 read_tbl = pd.read_csv(
    '../Liuetal_pipeline/pipeline/ribotrans_process/ribo_0h/ribotrans.csv.gz',
    usecols = ['codon_idx','gene','ribosome_count','TPM','codon']
    )

codonnum = pd.Series(range(0, len(CODON_TYPES)), index=CODON_TYPES)+1
n_cods = len(CODON_TYPES)+1 #There needs to be an uknown 
n_len = 512

#Index(['Unnamed: 0', 'gene', 'chrom', 'start', 'end', 'codon_idx',
#       'gene_strand', 'codon', 'pair_prob', 'TPM', 'ribosome_count'])

if True:
    print('parsing to tensors...')

    reads2use = (read_tbl.
                 query('codon_idx >= -10').
                 query('codon_idx < (512 -10 )').
                 query('ribosome_count!=0')
                 )

  


    reads2use = reads2use[reads2use.codon.isin(codonnum.index)]

    ugenes = reads2use.gene.unique()
    ugenes.sort()
    n_genes = ugenes.shape[0]
    gene2num = pd.Series(range(0, len(ugenes)), index=ugenes)
    
    assert reads2use.codon.isin(codonnum.index).all()

    # np.vstack([genenum[reads2use.gene].values,reads2use.codon_idx.values,codonnum[reads2use.codon].values]).shape
    poseffects = True

    ribosignal = torch.sparse.FloatTensor(
        torch.LongTensor([
            gene2num[reads2use.gene].values,
            reads2use.codon_idx.values+10
        ]),
        torch.FloatTensor(reads2use.ribosome_count.values),
        torch.Size([n_genes, n_len])).to_dense()
    
    TPMs = torch.log(ribosignal.mean(axis=1)**-1)

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

    assert torch.isfinite(TPMs).all()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def cget_batch(source, targetsource, offsetsource, i, bptt=100,device=device):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = targetsource[i:i+seq_len].unsqueeze(1)
        offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
        return data.to(device), target.to(device), offset.to(device)

    assert [x.shape for x in cget_batch(codons, ribosignal, TPMs, 1)] ==  [
         torch.Size([100, n_cods, 512 ]),
         torch.Size([100, 1, 512]),
         torch.Size([100, 1, 1])]


    print('splitting...')
    allinds = np.array(random.sample(range(0,n_genes),k=n_genes))
    traininds = allinds[0:int(n_genes*.6)]
    testinds = allinds[(int(n_genes*.6)):int(n_genes*.8)]
    valinds = allinds[int(n_genes*.8):]

    train_data, train_target, train_offsets = codons[traininds],ribosignal[traininds], TPMs[traininds]
    val_data, val_target, val_offsets = codons[valinds],ribosignal[valinds], TPMs[valinds]
    test_data, test_target, test_offsets = codons[testinds],ribosignal[testinds], TPMs[testinds]

    #we add these to the output of our model, to normalize for TPM  
