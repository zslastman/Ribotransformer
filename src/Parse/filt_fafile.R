library(tidyverse)
library(Biostrings)
fa='gencode.v24lift37.pc_translations.trid.fa'
ugenes= '../Liuetal_pipeline/pipeline/ribotrans_process/ribo_0h/ribotrans.csv.gz'
genes=ugenes%>%read_csv(col_types=cols_only(gene='c'))%>%unique
genes = genes[[1]]
filtfa = fa%>%str_replace('.fa$','.filt.fa')
seqs = Biostrings::readAAStringSet(fa)
namesnostop = vmatchPattern('*',seqs)%>%{names(.)[elementNROWS(.)==0]}
genes = intersect(namesnostop,genes)
seqs[genes]%>%writeXStringSet(filtfa)
