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
    args = parser.parse_args()

    model_file: Path = Path(args.m)
    fasta_file: Path = Path(args.f)
    el_mdl = torch.load(model_file)
    cdsob = Cdsannotation(fasta_file)
    coddf = cdsob.get_coddf(cdsob.trlength.index[0:4])
    # coddf = cdsob.get_coddf(lenfilt=10_000//3)
    Ixnosdata(coddf, countfilt=False)    


# for each chr

    # create the dataframe of codon info

    # now apply the model and get elongation scores

    # summarise these

    # save them to disk