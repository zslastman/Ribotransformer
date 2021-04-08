#!/usr/bin/bash

#first get teh translation using the file 

#Rscript /fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/src/Process/exp_cds_4trans.R --gtf=yeast_test/Yeast.sacCer3.sgdGene.gtf --fafile=yeast_test/Yeast.saccer3.fa --outprefix=../Ribotransformer/yeast_transcript.ext.fa --tpext=250 --fpext=250

conda init bash
conda activate pytorch
python Applications/esm/extract.py esm1_t34_670M_UR50S pipeline/yeast_transl.fa  ~/scratch/tmp/yeast_tokens    --include per_tok && echo 'finished!'
