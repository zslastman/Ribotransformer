################################################################################
########This script produces metacodon plots
################################################################################
library(rlang)
library(tidyverse)
library(GenomicRanges)
library(magrittr)
library(ggplot2)
library(riboWaltz)
library(data.table)
library(Rsamtools)

#TODOS
#remove the requirement for expression
library(R.utils)
args <- R.utils::commandArgs(trailingOnly = TRUE, asValues = TRUE,
    defaults = list(
        bamparentfolder = "../Liuetal_pipeline/pipeline/star/ORFext/data/", #folder with subfolders containing bam files
        fafile = '/fast/work/groups/ag_ohler/dharnet_m/Splicing_Lausanne/ext_data/annotation/gencode.v24lift37.annotation.orfext.fa',
        # riboexprfolder = '../Liuetal_pipeline/pipeline/ribotrans_process/',
        FLANKCODS = 14, 
        minreadlen=25,
        maxreadlen=35,
        offsetfile='../Liuetal_pipeline/pipeline/riboseqc/data/ribo_0h/_P_sites_calcs',
        outputfolder='../Liuetal_pipeline/pipeline/metacodon_offset_analysis',
        foo = 'bar'

))
# Turn arguments into R variables
keys <- attachLocally(args)
cat("Command-line arguments attached to global environment:\n");
print(keys);
str(mget(keys,  envir = globalenv()))

#utility functions
# base::source(str_interp::str_interp(paste0("../cortexomics/src/R/Rprofile.R")))


#the output folder
dir.create(outputfolder, recursive = TRUE, showWarnings = FALSE)

#folders with bam files
bfolders <- Sys.glob(paste0(bamparentfolder, "/*/"))
names(bfolders) <- basename(bfolders)
stopifnot(!anyDuplicated(names(bfolders)))

get_highcountcovtrs <- function(bfolders){
    bamfiles <- bfolders %>% map_chr(~Sys.glob(paste0(.,'*.bam')))
    names(bamfiles) <- names(bfolders)
    tr_read_stats = bamfiles%>%map_df(.id='sample',Rsamtools::idxstatsBam)
    tr_read_stats%>%group_by(sample)%>%mutate(mapped=mapped/sum(mapped))

    highcountcovtrs <- tr_read_stats %>%
        group_by(sample) %>%
        mutate(mapped=mapped/sum(mapped)) %>%
        group_by(seqnames) %>%
        summarise(m=mean(mapped)) %>%
        arrange(desc(m)) %>%
        dplyr::slice(1:5000)%>%
        pluck('seqnames')
    highcountcovtrs %<>% str_extract("^[^|]+")
}
message("choosing top 50000 transcripts")
if (!file.exists(str_interp("${outputfolder}/highcountcovtrs.rds"))) {
    #read the read data
    highcountcovtrs  <-  lapply(bfolders, get_highcountcovtrs)
    saveRDS(highcountcovtrs, str_interp("${outputfolder}/highcountcovtrs.rds"))
}else{
    highcountcovtrs <- readRDS(str_interp("${outputfolder}/highcountcovtrs.rds"))
}

highcountcovtrs <- get_highcountcovtrs(bfolders)

seqs <- readDNAStringSet(fafile)
seqheaders <- names(seqs)
names(seqs) <- str_extract(names(seqs), "^[^|]+")
headers_have_cds <- str_detect(seqheaders, "CDS:(\\d+)\\-(\\d+)")
stopifnot(headers_have_cds)
cdslens <- seqheaders %>%
    str_match("CDS:(\\d+)\\-(\\d+)") %>%
    .[, -1] %>%
    apply(2, as.numeric) %>%
    {.[, 2] - .[, 1] + 1}
seqlens <- nchar(seqs) %>% setNames(names(seqs))
names(cdslens) <- names(seqlens)

names(seqlens) %<>% str_extract("^[^|]+")

annotation_dt <- data.table(transcript = names(seqlens), l_utr5 = 60,
    l_cds = cdslens, l_utr3 = 57)%>%
    mutate(l_tr = l_utr5 + l_cds + l_utr3)
hctr_lens <- annotation_dt$l_tr[match(highcountcovtrs, annotation_dt$transcript)]
names(hctr_lens) <- highcountcovtrs

trspacecds <- annotation_dt %>%
    {GRanges(.$transcript, IRanges(
        .$l_utr5 + 1,
        .$l_utr5 + .$l_cds
    ))} %>% setNames(., seqnames(.))

# testtrs = annotation_dt

################################################################################
########Now read bam data
################################################################################
    
get_fpcovlist <- function(bfolder, hctr_lens, minreadlen, maxreadlen) {
        reads_list <- bamtolist(
                bamfolder = bfolder,
                annotation = annotation_dt)
        stopifnot(length(reads_list) == 1)
        reads_list = reads_list[[1]]
        GRanges(reads_list$transcript, IRanges(reads_list$end5, w = 1),
            readlen = reads_list$length) %>%
            subset(seqnames %in% highcountcovtrs) %>%
            subset(between(readlen, minreadlen, maxreadlen)) %>%
            { stopifnot(all(highcountcovtrs %in% seqnames(.) )) ;.} %>%
            { seqlevels(.) <- names(hctr_lens) ;.} %>%
            { seqlengths(.) <- hctr_lens ;.} %>%
            {split(., .$readlen)} %>%
            lapply(coverage)
}
get_fpcovlist <- purrr::partial(get_fpcovlist,hctr_lens=hctr_lens,
        minreadlen=minreadlen,
    maxreadlen=minreadlen)
#get allcodlist granges object descxribing codon positions in the transcripts
message("reading coverage from bam files")
if (!file.exists(str_interp("${outputfolder}/fpcovlist.rds"))) {
    #read the read data
    fpcovlist  <-  lapply(bfolders, get_fpcovlist)
    #
    fpcovlist%<>%setNames(names(bfolder))
    sampleorder = fpcovlist%>%names%>%str_extract("\\d+")%>%as.numeric
    stopifnot(!any(is.na(sampleorder)))
    stopifnot(!any(duplicated(sampleorder)))
    fpcovlist = fpcovlist[order(sampleorder)]
    saveRDS(fpcovlist, str_interp("${outputfolder}/fpcovlist.rds"))
}else{
    fpcovlist <- readRDS(str_interp("${outputfolder}/fpcovlist.rds"))
    stopifnot(allcodlist@seqinfo@seqnames %>% setequal(highcountcovtrs))
}

################################################################################
########Verify offsets with metacodon plots
################################################################################


i <- 1
allcodons <- names(Biostrings::GENETIC_CODE) %>% setNames(., .)
# seqlengths <- GenomicRanges::seqlengths
.="foo"


get_codon_gr <- function(codon, seqs, trspacecds,
    startbuff=60, endbuff=60, flankcods=FLANKCODS) {
    message(codon)
    #exclude the start ccodon
    codmatches <- Biostrings::vmatchPattern(pattern = codon, seqs)
    #
    matchgr <- codmatches %>% unlist %>% GRanges(names(.), .)
    cdsstarts <- start(trspacecds[as.vector(GenomicRanges::seqnames(matchgr))])
    matchgr$cdspos <- start(matchgr) - cdsstarts
    matchgr %<>% subset(cdspos %% 3 == 0)
    seqlengths(matchgr) <- setNames(nchar(seqs),names(seqs)) %>%
        .[seqlevels(matchgr)]
    innercds <- trspacecds %>%
        subset(width > (3 + startbuff + endbuff)) %>%
        resize(width(.) - startbuff, "end") %>%
        resize(width(.) - endbuff, "start")
    matchgr <- matchgr %>% subsetByOverlaps(innercds)
    codmatchwindows <- matchgr %>%
        resize(width(.) + (2 * (3 * flankcods)), "center")
    codmatchwindows <- codmatchwindows[!is_out_of_bounds(codmatchwindows)]
    codmatchwindows %<>% subsetByOverlaps(innercds)
    codmatchwindows
}

#get allcodlist granges object descxribing codon positions in the transcripts
if (!file.exists(str_interp("${outputfolder}/allcodlist.rds"))) {
    allcodlist <- lapply(allcodons, F = get_codon_gr,
        seqs = seqs[highcountcovtrs],
        trspacecds = trspacecds)
    allcodlist <- allcodlist %>% GRangesList %>% unlist
    saveRDS(allcodlist, str_interp("${outputfolder}/allcodlist.rds"))
}else{
    allcodlist <- readRDS(str_interp("${outputfolder}/allcodlist.rds"))
    stopifnot(allcodlist@seqinfo@seqnames %>% setequal(highcountcovtrs))
}


if (!file.exists(str_interp("${outputfolder}/fprustprofilelist.rds"))) {
    message("collecting RUST metacodon profiles")
    fprustprofilelist <- imap(fpcovlist, function(sampfpcov, sampname){
        message(sampname)
        #sum over counts for that transcript
        trsums <- sampfpcov %>% map(sum) %>% purrr::reduce(., `+`)
        sampfpcov %>% lapply(function(rlfpcov) {
            rlfpcov <- rlfpcov > mean(rlfpcov)
            nztrnames <- names(trsums)[trsums!=0]
            allcodlistnz <- allcodlist %>% subset(seqnames %in% nztrnames)
            cods <- names(allcodlistnz) %>% str_split("\\.") %>% map_chr(1)
            cat(".")
            out <- rlfpcov[allcodlistnz] %>%
                split(cods) %>%
                lapply(as.matrix) %>%
                map(colMeans)
            out
        })
    })
    fprustprofilelist <- fprustprofilelist[names(fpcovlist)]
    saveRDS(fprustprofilelist,str_interp("${outputfolder}/fprustprofilelist.rds"))
}else{
    fprustprofilelist<-readRDS(str_interp("${outputfolder}/fprustprofilelist.rds"))
}

process_profiles <- function(fprustprofilelist){
    rustprofiledat = fprustprofilelist%>%
        map_depth(3,.%>%enframe("position","count"))%>%
        map_df(.id="sample",.%>%
            map_df(.id="length",.%>%
                bind_rows(.id="codon")
            )
        )
    rustprofiledat%<>%mutate(position = position - 1 - (FLANKCODS*3))
    rustprofiledat%<>%group_by(sample,length,codon)%>%
        mutate(count= count / mean(na.rm=T,count))
    rustprofiledat%<>%filter(!codon %in% c("TAG","TAA","TGA"))
    rustprofiledat$length%<>%as.numeric
    rustprofiledat%<>%
        ungroup%>%
        mutate(sample = as_factor(sample))
    rustprofiledat
}
rustprofiledat <- process_profiles(fprustprofilelist)

#BUG
# rustprofiledat%>%filter(length==34,sample!="ribo_0h")

################################################################################
########testing the pca based a-site calls
################################################################################

get_metacodon_var_offsets<-function(rustprofiledat,outputfolder){
    profdat = rustprofiledat%>%select(position,length,sample,count)
    profdat$length= profdat$length%>%as.numeric
    profdat$phase = (-profdat$position)%%3
    vardf = profdat%>%
        ungroup%>%
        group_by(sample,length,position,phase)%>%
        summarise(sdsig=sd(count,na.rm=T)/median(count,na.rm=T))
    vardf <- 
        vardf%>%
        group_by(sample,length,phase)%>%
        arrange(position)%>%
        mutate(sdsigpair = sdsig+lag(sdsig))
    vardf <- vardf%>%
        mutate(ismode=(sdsig>lag(sdsig)) & ((sdsig)<lead(sdsig)))%>%
        filter(position> -length+6,position <  -6)
    bestmode <- vardf%>%group_by(sample,phase)%>%slice(which.max(sdsigpair))
    outvardf <- vardf%>%
        group_by(sample,length,phase)%>%
        slice(which.max(sdsigpair))%>%
        mutate(offset=-position)%>%
        select(length,phase,offset)
    #output text file
    varoffsetfile <- paste0(outputfolder,"/variance_offsets.txt")
    outvardf%>%write_tsv(varoffsetfile)
    varoffsetfile%>%
        normalizePath(mustWork=TRUE)%>%
        message
    outvardf
}
varoffsets <- get_metacodon_var_offsets(rustprofiledat,outputfolder)
#now plot the variation in occupancy at each position
#for each readlength
create_fp_var_plot <- function(rustprofiledat, varoffsets, outputfolder,
    plotname, ltrim=6, rtrim=6){
    plotfile <- paste0(outputfolder, plotname)
    grDevices::pdf(plotfile, w = 12, h = 12)
    #plotting variance amongst codons at each point.
    rustprofiledat %>% 
        ungroup %>% 
        group_by(sample, length, position) %>% 
        filter(!is.nan(count)) %>% 
        mutate(phase  =  -position %% 3) %>% 
        summarise(sdsig = sd(count, na.rm = T)  / mean(count, na.rm = T)) %>% 
        group_by(length, sample, position) %>% 
        summarise(sdsig = mean(sdsig)) %>% 
        filter(position > -length + ltrim, position < -rtrim) %>% 
        filter(length >= minreadlen, length <= maxreadlen) %>% 
        arrange(position) %>% {
            qplot(data = ., x = position, y = sdsig) +
            theme_bw() +
            facet_grid(length~as_factor(sample), scale = "free_y") +
            scale_y_continuous("between codon variation (meannorm)") +
            scale_x_continuous("5 read position relative to codon ") +
            geom_vline(data = varoffsets, aes(xintercept  =  -offset),
                color = I("blue"), linetype = 2) +
            geom_vline(data = varoffsets %>% mutate(offset  =  offset + 3),
                aes(xintercept =  -offset),
                color = I("red"), linetype = 2) +
            ggtitle("variance of 5' read occurance vs position")
        } %>% print
    dev.off()
    normalizePath(plotfile)%>%message
    normalizePath(plotfile)
}
for(sample_i in unique(rustprofiledat$sample)) {
    create_fp_var_plot(
        rustprofiledat %>% filter(sample==sample_i,length %>% between(27, 31)),
        varoffsets %>% filter(sample==sample_i,length %>% between(27, 31)) ,
        outputfolder,
        paste0("/", sample_i, "_trim_", "fppos_vs_codon_variance.pdf")
    )
    create_fp_var_plot(
        rustprofiledat %>% filter(sample==sample_i,length %>% between(27, 31)),
        varoffsets %>% filter(sample==sample_i,length %>% between(27, 31)) ,
        outputfolder,
        paste0("/", sample_i, "_notrim_", "fppos_vs_codon_variance.pdf"),
        ltrim = - 6,
        rtrim = - 6
    )
}



################################################################################
########Can also plot the pca to maybe seperate P and A site?
################################################################################

getpca <- function(pcadat, num=1) {
    pcadat %>%
    princomp %>%
    {.$loadings[, num] * .$sdev[num]} %>%
    enframe("position", "score")
}

#
profvarpca <- rustprofiledat %>%
    split(., .$sample) %>%
    map_df(.id = "sample", . %>%
        split(., list(.$readlen)) %>%
        map_df(.id = "length", function(profdata) {
            pcadat <- profdata %>%
                mutate(numreadlen = str_extract(readlen,"\\d+") %>% as.numeric) %>%
                filter(position > -numreadlen + 6, position < -6) %>%
                ungroup %>%
                select(-numreadlen, -readlen, -sample) %>%
                spread(position, count) %>%
                {set_rownames(.[, - 1], .$codon)}
            bind_rows(.id="pca",
                pca1 = pcadat %>% getpca(1),
                pca2 = pcadat %>% getpca(2),
                pca3 = pcadat %>% getpca(3)
            )
        })
    )
#
profvarpca %<>% select(sample, length, position, pca, score)
#Now plot the pcas vs the offsets
plotfile <- paste0(outputfolder, "/fppos_vs_codon_pcascore.pdf")
grDevices::pdf(plotfile, w = 12, h = 12);print(
profvarpca %>%
    ggplot(data = ., aes(y = score, x = as.numeric(position), color = pca)) +
    geom_point() +
    geom_line() +
    facet_grid(length~sample) + 
    geom_vline(data=varoffsets,aes(xintercept = -offset),
        color=I("blue"),linetype=2)+
    geom_vline(data=varoffsets%>%mutate(offset = offset+3),
        aes(xintercept= -offset),
        color=I("red"),linetype=2)
);dev.off()
normalizePath(plotfile)
