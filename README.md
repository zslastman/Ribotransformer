# Ribotransformer
The following text descibes the process to apply Ribotransformer, a transformer based tool for analyzing Ribosomal footprint density, to riboseq data.

The analysis is carried out in two steps. 

**Step 1:** An alignment file is processed to align riboseq footprints, and create a file with counts for each coding codon, as well as relevant CDS dimensions. This file will be used in Step 2 to run IxnosTorch. Genes are selected based on filtering criteria specified by the user.

**Step 2:** Using the read count files created in Step 1, A simple multilayer perceptron - the same algorithm as used by Ixnos - is used to create a local model of ribosome density accounting for cute site biases, codon dwell times etc.

**Step 3:** The model used in step3 is used as input to model long-range effects on ribosome density with Ribotransformer

All of the above steps can be run sequentially on multiple files using the Snakemake file ```ixnostrain.smk``` proved in the src folder


### Step 1:
The processbam file can be run like so:

```python src/processbam.py -i data/bamfile -f data/fafile -o data/ribotranstest```

The resulting files look like:

|tr_id	|codon_idx|	codon|	ribosome_count|
|:------:|:---------:|:------:|:---------:|
|	ENSMUST00000112172.3 |	-5 |	CTC |	0.0	|

Which indicate the transcript, the 0 indexed start of the start codon, index of each codon with the start codon at 0, the codon sequence, and the aligned A sites over that codon, respectively.


|tr_id	|aug|	stop|	length|
|:------:|:---------:|:------:|:---------:|
|	ENSMUST00000070533.4 |	150 |	2094 |	3634	|

Which indicate the transcript, the 0 indexed start of the start codon, the 0 indexed start of the stop codon, and the total transscript lenght, respectively.

### Step 2:

This step trains a model structurally identical (i.e. to the extent that weights learned in Ixnos can be used by it) to [Ixnos](https://github.com/lareaulab/iXnos), when used without RNA stucture information (which practically speaking I find to make little difference). The model is a multilayer perceptron that models local (+/- 15bp) influences on ribosome density. 

```
python src/train_ixmodel.py -i data/ribotranstest.all.psites.csv -c data/ribotranstest.cdsdims.csv -o data/ribotranstest
```

### Step 3:

This step trains a model of ribosome ribosome density that tries to capture long range interactions with an attention-based model. It uses the local model of ribosome density from step 2, and trims all CDS down to 512 basepairs by removing their middles to keep memory consumption reasonable.

```
python src/RiboTransData.py -i data/ribotranstest.all.psites.csv -c data/ribotranstest.cdsdims.csv -m data/ribotranstest.ixmodel_best.pt -o data/ribotranstest 
python src/ribotransformer.py -d data/ribotranstest_rdata.pt -o data/ribotranstest
```


