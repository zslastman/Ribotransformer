# base::source(here::here('src/Figures/Figure0/0_load_annotation.R'))
try(silent=T,{library(colorout)})
suppressMessages(library(R.utils))
library(tidyverse)
suppressMessages(library(magrittr))
suppressMessages(library(data.table))
suppressMessages(library(tidyverse))
suppressMessages(library(here))
suppressMessages(library(GenomicRanges))
suppressMessages(library(rtracklayer))
suppressMessages(library(Biostrings))
select <- dplyr::select


args = R.utils::commandArgs(trailingOnly=TRUE,asValues=TRUE,defaults =list(
  fasta='pipeline/yeast_transcript.ext.fa',
  outfile='pipeline/yeast_transcript.ext.pep.fa'
))

# Turn arguments into R variables
keys <- attachLocally(args)
cat("Command-line arguments attached to global environment:\n");
print(keys);
str(mget(keys, envir=globalenv()))
# }

seqs = readDNAStringSet(fasta)
snames = names(seqs)
snames = snames%>%str_split_fixed('\\|',12)
stopifnot(!any(duplicated(snames[,1])))
names(seqs) = snames[,1]
cdscoords = snames%>%.[,9]
cdscoords = cdscoords%>%str_match('(\\d+)\\-(\\d+)')

cdsgr = GRanges(names(seqs),IRanges(start=as.numeric(cdscoords[,2]),end=as.numeric(cdscoords[,3])))
cdsseq = seqs[cdsgr]
pepseq = cdsseq%>%translate
pepseq%>%writeXStringSet(outfile)
