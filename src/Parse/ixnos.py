# gene lengths file for yeast, or your own file with a custom transcriptome
import iXnos.interface as inter
expt_dir = "/fast/AG_Ohler/dharnet/Ribotransformer/iXnos/expts/lareau"
sam_fname = expt_dir + "/process/lareau.footprints.sam"
wsam_fname = inter.edit_sam_file(
    expt_dir, sam_fname, filter_unmapped=True, 
    sam_add_simple_map_wts=False, RSEM_add_map_wts=True)

gene_len_fname = '/fast/AG_Ohler/dharnet/Ribotransformer/iXnos/genome_data/scer.transcripts.13cds10.lengths.txt'
inter.do_frame_and_size_analysis(
    expt_dir, wsam_fname, gene_len_fname, min_size=13,
    max_size=36, verbose=False)
