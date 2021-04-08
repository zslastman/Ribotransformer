library(Rsamtools)
source('../cortexomics/src/R/Rprofile.R')
seqs = '../cortexomics/yeast_test/saccer3.fa'%>%Biostrings::readDNAStringSet(.)
cdsseqs = trs%>%subset(type=='CDS')%>%split(.,.$transcript_id)%>%.[str_order_grl(.)]%>%GenomicFeatures::extractTranscriptSeqs(FaFile('../cortexomics/yeast_test/saccer3.fa'),.)

cdsseqs%>%writeXStringSet('yeast_transl.fa')