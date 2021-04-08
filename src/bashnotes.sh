cact pytorch
python Applications/esm/extract.py esm1_t34_670M_UR50S yeast_transl.fa  ~/scratch/tmp/yeast_tokens    --include per_tok && echo 'finished!'


Rscript /fast/work/groups/ag_ohler/dharnet_m/Ribotransformer/src/Process/exp_cds_4trans.R --gtf=yeast_test/Yeast.sacCer3.sgdGene.gtf --fafile=yeast_test/Yeast.saccer3.fa --outprefix=../Ribotransformer/yeast_transcript.ext --tpext=250 --fpext=250

STAR --runThreadN 8 --runMode genomeGenerate --genomeDir tr_index --genomeFastaFiles yeast_transcript.ext.shortheader.fa --genomeSAindexNbases 11 --genomeChrBinNbits 12

nmismatch=1
SAM_params="--outSAMtype BAM Unsorted --outSAMmode NoQS --outSAMattributes NH NM "
align_params="--seedSearchLmax 10 --outFilterMultimapScoreRange 0 --outFilterMultimapNmax 255 --outFilterMismatchNmax ${nmismatch} --outFilterIntronMotifs RemoveNoncanonical --sjdbOverhang 35"
ribofastq=/fast/work/groups/ag_ohler/dharnet_m/cortexomics/yeast_test/input/srr104951/SRR1049521.trim.fastq.gz
rnafastq1=/fast/work/groups/ag_ohler/dharnet_m/cortexomics/yeast_test/input/srr1049520/SRR1049520.trim.fastq.gz
STAR --runThreadN 8 --genomeDir tr_index --readFilesIn <(zcat  $ribofastq ) --outFileNamePrefix 'weinberg_yeast_ribo' ${SAM_params} ${align_params}
STAR --runThreadN 8 --genomeDir tr_index --readFilesIn <(zcat  $rnafastq ) --outFileNamePrefix 'weinberg_yeast_rna' ${SAM_params} ${align_params}


