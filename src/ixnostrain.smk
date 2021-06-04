import pandas as pd

bamfiles = pd.read_csv('bamlist.tsv', sep='\t').bamfile
bamfiles = bamfiles
samples = list(bamfiles.index)

# import ipdb; ipdb.set_trace()
offsetops={}
fafiles={}
for sample in samples:
	bam = bamfiles[sample]
	if 'cortexomics' in bam:
		offsetops[sample]=' -l ../../cortexomics/ext_data/offsets_manual.tsv'
		fafiles[sample]='../../cortexomics/ext_data/gencode.vM12.pc_transcripts.fa'

rule all:
	input:
		expand('ixnosmodels/{sample}/{sample}', sample=samples)

rule get_read_df:
    input: bamfile = lambda wc: bamfiles[wc.sample], fafile = lambda wc: fafiles[wc.sample]
    params: offset = lambda wc: offsetops[wc.sample]
    output: 
    	touch('ribotransdata/{sample}/{sample}'),
    	'ribotransdata/{sample}/{sample}.all.psites.csv',
    	'ribotransdata/{sample}/{sample}.cdsdims.csv'
    shell: r'''
	python ../src/IxnosTorch/processbam2.py -i {input.bamfile} -f {input.fafile} {params.offset} -o {output[0]}
	'''

rule train_ixmodel:
    input:
        readdata = 'ribotransdata/{sample}/{sample}.all.psites.csv',
        cdsdims = 'ribotransdata/{sample}/{sample}.cdsdims.csv'
    output: touch('ixnosmodels/{sample}/{sample}')
    shell: r'''
	python ../src/IxnosTorch/train_ixmodel.py -i {input.readdata} -c {input.cdsdims} -o {output}
	'''

rule ixnos_elong:
    input:
        model = 'ixnosmodels/{sample}/{sample}.bestmodel.pt',
        fafile = lambda wc: fafiles[wc.sample]
    output: touch('ixnos_elong/{sample}/{sample}'),'ixnos_elong/{sample}/{sample}.elong.tsv'
    shell: r'''
	python ../src/IxnosTorch/ixnos_elong.py -m {input.model} -f {input.fafile} -o {output}
	'''