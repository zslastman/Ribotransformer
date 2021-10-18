#!/usr/bin/bash

#this script uses https://github.com/facebookresearch/esm
#to get evolution-informed representations of ORFs, for later use in Ribotransformer

source ~/work/miniconda3/etc/profile.d/conda.sh
#first get teh translation using the file 
#Rscript /fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/src/Process/exp_cds_4trans.R --gtf=yeast_test/Yeast.sacCer3.sgdGene.gtf --fafile=yeast_test/Yeast.saccer3.fa --outprefix=../Ribotransformer/yeast_transcript.ext.fa --tpext=250 --fpext=250
#Rscript /fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/src/Process/make_peptide_fasta.R yeast_transcript.ext.pep.fa --outfile=../Ribotransformer/yeast_transcript.ext.fa
conda activate pytorch
python Applications/esm/extract.py esm1_t34_670M_UR50S ../Ribotransformer/pipeline/yeast_transcript.ext.pep.fa  ~/scratch/tmp/yeast_tokens    --include per_tok && echo 'finished!'
