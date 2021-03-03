
# offsets = offsetfile %>%
#      fread %>%
#      filter(comp=='nucl') %>%
#      select(length = read_length, comp, offset = cutoff)
# # offsets %<>% mutate(readlen = paste0(length))
# offsets %<>% filter(length>=minreadlen,length>=maxreadlen) 



riboexpr = riboexprfolder %>%
    {Sys.glob(paste0(., "*/ribotrans_expr.tr_expr.tsv"))} %>%
    setNames(., basename(dirname(.))) %>%
    map_df(.id = "sample", fread)
riboexpr%<>%mutate(msample = sample %>% str_replace("ribo", "r"))


get_highcountcovtrs <- function(riboexpr){
    riboexpr %>%
        filter(cds_len > 100) %>%
        group_by(tr_id)%>%summarise(fdens = mean(RPF_dens)) %>%
        arrange(desc(fdens)) %>%
        slice(1:5000) %>%
        .$tr_id
}