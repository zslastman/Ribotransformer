#!/usr/bin/env python

## ----------------------------------------
## scikit-ribo 
## ----------------------------------------
## a module for processing bam files
## ----------------------------------------
## author: Han Fang 
## contact: hanfang.cshl@gmail.com
## website: hanfang.github.io
## date: 10/28/2016
## ----------------------------------------

from __future__ import print_function, division
import os
import sys
import argparse
import pybedtools as pbt
import pysam
import pandas as pd
import numpy as np
import csv
import errno
import pyfastx
import ipdb
import pyranges as pr
import pandas as pd
import numpy as np
import pyranges as pr
import textwrap
import time
from Bio.Seq import Seq

from datetime import datetime


def add_csum(trs):
    assert 'transcript_id' in trs.columns
    assert isinstance(trs, pr.PyRanges)
    trs = (trs.
           as_df().
           assign(width=lambda x: x.End - x.Start).
           assign(
               csum=lambda df:
               df.groupby('transcript_id')['width'].
               apply(lambda x: x.cumsum()-x),
               rcsum=lambda df:
               df.groupby('transcript_id')['width'].
               apply(lambda x: x.cumsum()-x),
           ))
    return pr.PyRanges(trs)


def trlens(annogrl):
    return (
        annogrl.as_df().assign(width=lambda x: x.End - x.Start).
        groupby('transcript_id')['width'].sum()
        )


def resize_trs_startfix(pyr_orig, newwidth, alter=False):
    assert pyr_orig.length > 0
    pyr = pyr_orig.copy()
    if(alter):
        newwidth = trlens(pyr) + newwidth
    assert (newwidth > 0).all()
    # get the cumulative length of tr at the end of each exon
    pyr = pyr.as_df().assign(width=lambda x: x.End - x.Start)
    csum = (pyr.
            groupby('transcript_id')['width'].
            apply(lambda x: x.cumsum()-x)
            )
    drop = csum > newwidth[pyr.transcript_id].values
    trim = csum + pyr.width - newwidth[pyr.transcript_id].values
    pyr = pyr.drop(['width'], 1)
    # now drop the ones we need to
    pyr = pyr[~drop]
    trim = trim[~drop]
    # calculate which ranges are 3' ends
    is_tpend = (trim.
                groupby(pyr.transcript_id.values).
                apply(lambda x: x == x.max()))
    is_neg = pyr.Strand == '-'
    # and trim the 3' ends by the necessary amount
    postp = (is_tpend.values & (~is_neg.values))
    pyr.End = np.where(postp, pyr.End.values - trim.values, pyr.End)
    # and for negative ranges
    negtp = (is_tpend.values & (is_neg.values))
    pyr.Start = np.where(negtp, pyr.Start + trim.values, pyr.Start)
    return pr.PyRanges(pyr)
#


def sort_pyr_st(pyr):
    return (
        pyr.
        assign('order', lambda x: np.where(x.Strand == '-', -1, 1)*x.Start).
        sort(['Chromosome', 'Strand', 'order']).
        drop(['order'])
        )
#


def invert_strand(pyr):
    pyr = pyr.as_df()
    pyr.Strand = pd.Series(np.where(pyr.Strand == '-', '+', '-'))
    return pr.PyRanges(pyr)
#


def resize_trs_endfix(pyr, newwidth, alter=False):
    pyr = sort_pyr_st(invert_strand(pyr))
    pyr = resize_trs_startfix(pyr, newwidth, alter)
    return sort_pyr_st(invert_strand(pyr))
#


def resize_trs(pyr, newwidth, fix='start', alter=False):
    if type(newwidth) == int:
        newwidth = pd.Series(newwidth, index=pyr.transcript_id.unique())
    assert pyr.as_df().groupby(['Chromosome', 'Strand']).apply(lambda x: (
        np.diff(np.where(x.Strand == '-', -1, 1)*x.Start) >= 0).all()).all()
    #
    if fix == 'start':
        pyr = resize_trs_startfix(pyr, newwidth, alter)
    elif fix == 'end':
        pyr = resize_trs_endfix(pyr, newwidth, alter)
    elif fix == 'center':
        diffs = newwidth if(alter) else trlens(pyr)+newwidth
        #
        pyr = resize_pyr_startfix(pyr, ceiling(diffs/2), alter=True)
        pyr = resize_pyr_endfix(pyr, diffs, alter, True)
    else:
        ValueError("fix needs to be one of start, end, center")
    return pyr

def get_codons(cdspyr, cdsseq):
    adjpyr = cdspyr.copy()
    adjpyr = add_csum(adjpyr)
    adjpyr = sort_pyr_st(adjpyr)
    adjpyr.phase = ((3 - adjpyr.csum) % 3)
    # clip the 5' ends
    adjpyr=adjpyr.as_df()
    ispos = adjpyr.Strand == '+'
    isneg = ~ ispos
    adjpyr.loc[ispos,'Start'] = adjpyr.Start[ispos] + adjpyr.phase[ispos]
    adjpyr.loc[isneg,'End'] = adjpyr.End[isneg] - adjpyr.phase[isneg]
    # extend the 3' ends
    adjpyr = adjpyr.assign(width=lambda x: x.End-x.Start)
    ext = (-(adjpyr.width)%3)
    adjpyr.loc[ispos, 'End'] = adjpyr.End[ispos] + ext[ispos]
    adjpyr.loc[isneg, 'Start'] = adjpyr.Start[isneg] - (ext[isneg])

    n_codexts = adjpyr.groupby('transcript_id').head(1).set_index('transcript_id').fpext
    trstrands = adjpyr.groupby('transcript_id').head(1).set_index('transcript_id').Strand
    #
    codons = []
    for tr,exons in adjpyr[['Chromosome','transcript_id','Start','End']].groupby('transcript_id'):
        trdf = []
        trstand = trstrands[tr]
        for i,chr,tr,st,end in exons.itertuples():    
            ncods = int((end-st)/3)
            if(trstand=='+'):
                trdf.append( 
                  pd.DataFrame(
                    list(zip(
                    [tr]*ncods,
                    [chr]*ncods,
                    range(st,end-1,3),
                    range(st+3,end+2,3)
                  )),columns=['transcript_id','Chromosome','Start','End'])
                )
            else:
                trdf.append( 
                  pd.DataFrame(
                    list(zip(
                    [tr]*ncods,
                    [chr]*ncods,
                    range(end-3,st-3,-3),
                    range(end,st,-3)
                  )),columns=['transcript_id','Chromosome','Start','End'])
                )
        n_codext = n_codexts[tr]
        codex = int(n_codext/3)
        trdf = pd.concat(trdf)
        codrange = range(-codex,int(trdf.shape[0])-codex)
        trdf = trdf.assign(codon_idx=list(codrange))
        trdf = trdf.assign(strand=trstand)
        trdf = trdf.assign(codon=textwrap.wrap(cdsseq[tr],3))
        codons.append(trdf)
    #and combine
    codons = pd.concat(codons)
    codonsrnm = codons.rename(columns={'transcript_id':'gene','Chromosome':'chrom','Start':'start','End':'end','strand':'gene_strand'})
    codonsrnm[['chrom','start','end','gene','codon_idx','gene_strand','codon']]
    # codonsrnm[codonsrnm.codon_idx==0]
    return codonsrnm 



def load_anno(gtf, fasta, n_codonext):
    # gtffile = 'Yeast.sacCer3.sgdGene.gtf'
    prgtf = pr.read_gtf(gtf)
    prgtf = sort_pyr_st(prgtf)
    cdspyr = add_csum(prgtf[prgtf.Feature == 'CDS'])
    exonswcs = add_csum(prgtf[prgtf.Feature == 'exon'])
 
    #we can't extend past chromosome start
    assert (cdspyr.Start > n_codonext).all
    assert hasattr(cdspyr,'transcript_id')
    assert cdspyr.transcript_id.isin(exonswcs.transcript_id).all()
    exonswcs = exonswcs[exonswcs.transcript_id.isin(cdspyr.transcript_id)]


    # assert(((cdslens % 3)==0).all())
    cdslens = trlens(cdspyr)
    if(not ((cdslens % 3)==0).all()):
        is3bp = (cdslens % 3) ==0
        n_not3bp = (~is3bp).sum()
        print('removing '+str(n_not3bp)+' ORFs')
        exonswcs = exonswcs[is3bp[exonswcs.transcript_id]]
        cdspyr = cdspyr[is3bp[cdspyr.transcript_id]]

    cdsstarts = (
        cdspyr[(cdspyr.csum == 0)][["transcript_id"]].
        sort().
        join(exonswcs[["csum", "transcript_id"]]).
        subset(lambda x: x.transcript_id_b == x.transcript_id).
        assign('cdsstart', lambda x: pd.Series(
            np.where(
                x.Strand == '-',
                x.End_b - x.End + x.csum,
                x.Start - x.Start_b + x.csum
            )))
        ).as_df().set_index('transcript_id').cdsstart
    
    exonlens = trlens(exonswcs)
    cdslens = trlens(cdspyr)

    cdsends = cdsstarts+cdslens
    threeext = (n_codonext - (exonlens - cdsends)).astype(np.int32)
    fiveext = (n_codonext - cdsstarts).astype(np.int32)

    assert (cdslens <= exonlens + fiveext + threeext).all()
    assert (exonlens + fiveext + threeext < (1+cdslens+n_codonext*2)).all()

    print("creating extended cds...")
    cdsext = resize_trs(exonswcs, fiveext, fix='end', alter=True)
    cdsext = resize_trs(cdsext, threeext, fix='start', alter=True)
    cdsext = add_csum(cdsext)
    cdsext = sort_pyr_st(cdsext)
    assert (trlens(cdsext) == exonlens + fiveext + threeext).all()

    #record the extensions made - trim if necessary
    cdsext.ext = n_codonext
    adj = ((cdsext.Start*-1)/3).apply(np.ceil).mul(3).apply(np.int32)
    adj = np.where(cdsext.Start < 0,adj,0)
    cdsext.Start += adj
    cdsext.fpext = n_codonext-adj
    cdsext.Start = np.int32(cdsext.Start)

    #now get the fasta chromosomes
    assert os.path.isfile(fasta)
    seqchrs = os.popen("grep -e '>' "+fasta).read().replace('>','').splitlines()
    seqchrs = [s.split(' ')[0] for s in seqchrs]
    print('got chromosomes from fasta like...' + seqchrs[0])
    assert cdsext.Chromosome.isin(seqchrs).all()
    # fasta = 'Yeast.saccer3.fa'
    print("getting fasta sequence for the extended cds...")
    extcdsseq = pr.get_fasta(cdsext.unstrand(), fasta)
    outofboundtrs = cdsext.transcript_id[extcdsseq=='']
    extcdsseq = extcdsseq[~ cdsext.transcript_id.isin(outofboundtrs).values]
    cdsext = cdsext[~ cdsext.transcript_id.isin(outofboundtrs).values]

    seqdf = pd.concat([pd.Series(extcdsseq.values, name='seq'),
                       cdsext.transcript_id.reset_index(drop=True),
                       cdsext.Strand.reset_index(drop=True)], axis=1)
    seqdf.seq = [str(Seq(se).reverse_complement())
                 if st is '-' else se for i, se, tr, st in seqdf.itertuples()]
    cdsseq = seqdf.groupby('transcript_id')['seq'].apply(lambda s: ''.join(s))

    cdsext.transcript_id = cdsext.transcript_id.str.replace(';','').values
    cdsseq.index = cdsseq.index.str.replace(';','').values

    extlens = trlens(cdsext)
    extlens = extlens[cdsseq.index]

    assert (extlens == cdsseq.str.len()).all()
    codons = get_codons(cdsext, cdsseq)
    n_cds = str(cdsseq.shape[0])
    print("successfully got "+n_cds+" cds from our annotation")

    return cdsext, cdsseq, codons

fasta='/fast/work/groups/ag_ohler/dharnet_m/cortexomics/yeast_test/Yeast.saccer3.fa'
gtf = '/fast/work/groups/ag_ohler/dharnet_m/cortexomics/yeast_test/Yeast.sacCer3.sgdGene.gtf'
cdsext,cdsseq,codons = load_anno(gtf, fasta, 0)


# # the main process
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-g", help="gtf file, required")
#     parser.add_argument("-r", help="fasta file, required")
#     parser.add_argument("-p", help="prefix to use, required")
#     parser.add_argument("-o", help="output path, required")
#     ## check if there is any argument
#     if len(sys.argv) <= 1: 
#         parser.print_usage() 
#         sys.exit(1) 
#     else: 
#         args = parser.parse_args()
#     ## process the file if the input files exist
#     if (args.g!=None) & (args.r!=None) & (args.o!=None) & (args.p!=None):
#         sys.stderr.write("[status]\tReading the input file: " + args.g + "\n")
#         gtf = args.g
#         ref = args.r
#         prefix = args.p
#         output = args.o
#         # create output folder
#         cmd = 'mkdir -p ' + output
#         os.system(cmd)
#         ## execute
#         sys.stderr.write("[execute]\tStarting the pre-processing module" + "\n")
#         worker = GtfPreProcess(gtf, ref, prefix, output)
#         sys.stderr.write("[execute]\tLoading the the gtf file in to sql db" + "\n")
#         worker.convertGtf()
#         sys.stderr.write("[execute]\tCalculating the length of each chromosome" + "\n")
#         worker.getChrLen()
#         sys.stderr.write("[execute]\tExtracting the start codons' positions from the gtf db" + "\n")
#         worker.getStartCodon()
#         sys.stderr.write("[execute]\tExtracting the sequences for each gene" + "\n")
#         worker.getSeq()
#         sys.stderr.write("[execute]\tBuilding the index for each position at the codon level" + "\n")
#         worker.getCodons()
#         worker.getNts()
#         sys.stderr.write("[execute]\tCreating the codon table for the coding region" + "\n")
#         worker.createCodonTable()
#         sys.stderr.write("[status]\tGtf processing module finished" + "\n")
#     else:
#         sys.stderr.write("[error]\tmissing argument" + "\n")
#         parser.print_usage() 
