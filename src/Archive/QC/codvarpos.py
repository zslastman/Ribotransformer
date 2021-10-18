


#Now for each codon, get the variance in the locations of 5' read
#read positions

codons#matrix of codon positions
offsetrange#range of offsets to check.
rldata#gene,position,readlength tensor


#pytorch function roll.
#my own functiont the arolls with no wraparound?

#inputs to this program - bam file, and and the fasta it was aligned to, along with
#cds coordinates - either in the fasta or a seperate table.

#turn the reads into a big nucleotide res tensor.
#One could build this thing as one goes along the reads, but maybe just as
#easy to make it from my data read data frame.


#In the case of the yeast data, we can use the hdf5 data
#We can't use the yeast test data as is now, cos it's already
#codon collapsed - no good.

#so for yeast data, parse the hdf5 data into a read dataframe in
#transcript space
#making sure it all lines up,
#then pair with the codon data which we've gotten from ...
#a protein coding fasta file
#But will this protein coding fasta file match up with the 
#perhaps make use of RUST?

#I could possibly make code to load genomic alignments if I wanted...
#Fuck that.



