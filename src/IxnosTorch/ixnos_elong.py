###############################################################################
########
###############################################################################
	
import torch
import argparse
# import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from seqfold import dg
from pathlib import Path


from train_ixmodel import Net
from ixnosdata import Ixnosdata
from processbam import Cdsannotation

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ixnosdata", "/fast/AG_Ohler/dharnet/Ribotransformer/src/IxnosTorch/ixnosdata.py")
ixnosdata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ixnosdata)
Ixnosdata = ixnosdata.Ixnosdata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        help="model object for elongation calculations",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default=f'ixnosmodels/E13_ribo_1/E13_ribo_1.bestmodel.pt'
    )
    parser.add_argument(
        "-f",
        help="cds fasta file to get elongations for",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default=f'../../cortexomics/ext_data/gencode.vM12.pc_transcripts.fa'
    )
    parser.add_argument(
        "-o",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        help=f'the stem of the output file the program creates',
        default=f'ixnos_elong/E13_ribo_1/E13_ribo_1'
    )
    args = parser.parse_args()
 
    model_file: Path = Path(args.m)
    fasta_file: Path = Path(args.f)
    outputfile: Path = Path(f'{args.o}.elong.csv')

    el_mdl = Net.from_state_dict(torch.load(model_file)['beststate'])

    cdsob = Cdsannotation(fasta_file)
    cdsdims = cdsob.cdsdims

    # coddf = cdsob.get_coddf(lenfilt=10_000//3)
    # assert len(ixdata.traintrs)== len(ixdata.bounds_tr)

    # chunk our transcripts so we don't end up with too much data at once
    ncodchunk = cdsdims.sort_values('n_cod').set_index(
        'tr_id')['n_cod'].cumsum().divide(100_000).round()
    ncodchunk = ncodchunk.reset_index().groupby('n_cod')['tr_id']
    # now iterate over these
    allpreds=[]
    i=0
    for n_cod, chunktrs in ncodchunk:
        # get the codon df for these trs
        coddf = cdsob.get_coddf(chunktrs,rflank=1,lflank=1)
        # convert into an ixnos object (1 hot encoded matrix)
        ixdata = Ixnosdata(coddf, cdsdims,  tr_frac=1,
                           fptrim=5, tptrim=5)
        # get the cumulative bounds of these from the sizes
        cbounds = np.cumsum([0]+ixdata.bounds_tr)

        # now get predictions with our model
        predictions = el_mdl(ixdata.X_tr.float()).detach().numpy()

        # using the bounds, chunk and average our predictions
        chunkpredlist = [(tr, predictions[l:r].mean()) for tr, l, r in zip(
            ixdata.traintrs, cbounds, cbounds[1:])]
        # as a series
        chunkpredsr = pd.Series(dict(chunkpredlist))
        allpreds.append(chunkpredsr)
        # print(f'size of data object: {ixdata.X_tr.shape[0]}')
        # i+=1
        # if i > 1: break
    # combine
    allpredscat: pd.DataFrame = pd.concat(allpreds).reset_index()
    allpredscat.columns = ['tr_id', 'elong']
    # and output
    print(outputfile)
    allpredscat.to_csv(outputfile, index=False)
    


# for each chr

    # create the dataframe of codon info

    # now apply the model and get elongation scores

    # summarise these

    # save them to disk
