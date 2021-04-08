# # Janggu For Riboseq

# Project priorities
#
# - Get basic run of everything
#     - Run convnet
#     - Run LSTM
#     - Run transformer
#     - Use yeast data for the above
#
# - Try to get Janggu running somehow.
#     - Janggu for pulling Riboseq
#     - Janggu for doing the tokens
#     - Janngu just in general, for pulling the sequences I need.
#     
# - Parts of this data loading
#     - Coverage data
#         - Not that big, should be easy to hdf5
#         - possibly needs to be able to load with a dictionary of offsets
#     - Sequence data
#         - again okay, but should probably be loaded with janggu
#         - easy to load this using tr fasta files and cds coords
#         - I can get codons too, which I guess I slice with ::3
#             - But not sure about padding - what do I do with different lengths of cds?
#     - tokens
#         - janggu isn't really suited for this I think? I think I'd have to load everything into one array before I could do anything.
#      
# Goals:
#     - as discussed with wolfgang, try creating a hdf5 from genomic array
#     - establish how ot deal with different length trs
#     
#

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas
import janggu
import torch
import numpy as np
from janggu.data import Bioseq
from janggu.data.coverage import Cover

# +
fastafile = '/fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/pipeline/yeast_transcript.ext.fa'
cdscoords = '/fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/pipeline/yeast_transcript.ext_trspaceanno.gtf'
bedcols =['chrom', 'start', 'end','name', 'score', 'strand']
gtfcols = ['chrom','source','type', 'start', 'end',
                                 'score','strand','frame','attribute']
roi = pandas.read_csv(cdscoords, sep='\t', header=None, names=gtfcols,skiprows=3)
roi = roi.query('type=="CDS"')
roi['strand']='+'
roi['start']=roi['start']-1

roi
# -

dict = {28:2}
#dict = None
offset = dict.get(29) if dict else 0


DNA = Bioseq.create_from_refgenome('yeastseq',
                             '/fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/pipeline/yeast_transcript.ext.shortheader.fa',
                             roi = roi,order=3)

DNA.shape
DNA[0].shape
acgt=np.array(['A','C','G','T'])
npseq = torch.from_numpy(DNA[0:20])
npseq.shape
np.argmax(npseq[:,2,0],axis=1)
#checks that we pull ATG from the start
#[ ''.join([str(acgt[np.where(npseq[s,n,0,:])][0]) for n in range(0,4)]) for s in range(0,len(npseq))]
#[ [np.where(npseq[s,n,0,:]) for n in range(0,4)] for s in range(0,4)]
#[acgt for n in npseq[s,ntorch.from_numpy(DNA[0:4])[:,:3].shape



bamfile_ = '/fast/work/groups/ag_ohler/dharnet_km/cortexomics/yeast_test/weinberg_yeast_riboAligned.out.sort.bam'

gtf = '/fast/work/groups/ag_ohler/dharnet_m/cortexomics/yeast_test/Yeast.sacCer3.sgdGene.gtf'

bedcols =['chrom', 'start', 'end','name', 'score', 'strand']
gtfcols = ['chrom','source','type', 'start', 'end',
                                 'score','strand','frame','attribute']
roi = pandas.read_csv(gtf, sep='\t', header=None, names=gtfcols)

Bioseq.create_from_refgenome('yeastseq',fastafile='/fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/pipeline/yeast_transcript.ext.fa')

roi['name'] = '.'
roi[['chrom', 'start', 'end','name', 'score', 'strand']].to_csv('roi.bed',header=None,sep='\t')
roi = pandas.read_csv('roi.bed',sep='\t',header=None,names=bedcols)

roi = roi.head(100)
roi

:store = ''
swg=False
cov = Cover.create_from_bam(
            'test',
            bamfile_,
            resolution=1,
            binsize=20,
            cache=True,
            roi=roi, zero_padding=False,
            stranded=False,
            storage='ndarray',
            store_whole_genome=swg)

cov.shape

cov[0,0,20].shape

# %run /fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/Applications/janggu/src/janggu/data/coverage.py

store = ''
swg=False
cov = Cover.create_from_bam(
            'test',
            bamfile_,
            resolution=1,
            binsize=300,
            cache=True,
            roi=roi, zero_padding=False,
            stranded=False,
            storage='hdf5',
            store_whole_genome=swg,
            readlens=[27,28,29,30])

roi

cov[0,]

cov[0,2,100].sum(axis=(1,2,3)).shape

cov

cov[1,]

cov.loader

cov


