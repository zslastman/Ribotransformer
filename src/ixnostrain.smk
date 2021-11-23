import pandas as pd

config['bamlist']
if config.get('bamlist') is None:
	bamlist = config['bamlist']
	bamfiles = pd.read_csv(bamlist, sep='\t').bamfile
else:
	bamlist = config['bamlist']
	bamfiles = pd.Series(bamlist)

samples = list(bamfiles.index)

# import ipdb; ipdb.set_trace()
offsetops={}
fafiles={}
offsetops.update(config['offsetops'])
fafiles.update(config['fafiles'])

for sample in samples:
	bam = bamfiles[sample]
	# if 'cortexomics' in bam:
		# offsetops[sample]=' -l ../../cortexomics/ext_data/offsets_manual.tsv'
		# fafiles[sample]='../../cortexomics/ext_data/gencode.vM12.pc_transcripts.fa'
	assert fafiles[sample]
	assert bamfiles[sample]


rule all:
	input:
		expand('ixnosmodels/{sample}/{sample}', sample=samples),
		expand('ixnos_elong/{sample}/{sample}', sample=samples)

rule get_read_df:
    input: bamfile = lambda wc: bamfiles[wc.sample], fafile = lambda wc: fafiles[wc.sample]
    params: offset = lambda wc: ' -l '+offsetops[wc.sample]  if offsetops[wc.sample] else '' 
    output: 
    	touch('ribotransdata/{sample}/{sample}'),
    	'ribotransdata/{sample}/{sample}.all.psites.csv',
    	'ribotransdata/{sample}/{sample}.cdsdims.csv'
    shell: r'''
	python src/processbam.py -i {input.bamfile} -f {input.fafile} {params.offset} -o {output[0]}
	'''

rule train_ixmodel:
    input:
        readdata = 'ribotransdata/{sample}/{sample}.all.psites.csv',
        cdsdims = 'ribotransdata/{sample}/{sample}.cdsdims.csv'
    output: touch('ixnosmodels/{sample}/{sample}'),'ixnosmodels/{sample}/{sample}.bestmodel.pt'
    shell: r'''
	python src/train_ixmodel.py -i {input.readdata} -c {input.cdsdims} -o {output[0]}
	'''

rule ixnos_elong:
    input:
        model = 'ixnosmodels/{sample}/{sample}.bestmodel.pt',
        fafile = lambda wc: fafiles[wc.sample]
    output: touch('ixnos_elong/{sample}/{sample}'),'ixnos_elong/{sample}/{sample}.elong.csv'
    shell: r'''
	python src/ixnos_elong.py -m {input.model} -f {input.fafile} -o {output[0]}
	'''


rule ribotrans_elong:
    input:
        readdata = 'ribotransdata/{sample}/{sample}.all.psites.csv',
        cdsdims = 'ribotransdata/{sample}/{sample}.cdsdims.csv'
        model = 'ixnosmodels/{sample}/{sample}.bestmodel.pt',
       # fafile = lambda wc: fafiles[wc.sample]
    output: 
    	touch('ribotrans_elong/{sample}/{sample}'),
    	'ribotrans_elong/{sample}/{sample}_rdata.pt',
    	'ribotrans_elong/{sample}/{sample}_ribotrans_ix_elongs.csv',
    shell: r'''
	python src/RiboTransData.py -i {input.readdata} -c {input.cdsdims} -m {input.model} -o ${output[0]} 
	python src/ribotransformer.py -d {output[0]}_rdata.pt -o {output[0]}
	'''