import pysam
import sys
import textwrap
import multiprocessing
import numpy as np
import pandas as pd
import itertools as it
import pybedtools as pbt
from textwrap import wrap
from scipy import sparse
# from scikit_ribo import asite_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

# from sklearn import preprocessing, svm, tree
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectFromModel, RFECV
# import glmnet_py.dataprocess as dataprocess
import_folder = '/fast/work/groups/ag_ohler/dharnet_m/cortexomics/src/Ribotransformer/'
sys.path = [import_folder]+list(set(sys.path)-set(import_folder)) # this tells python to look in `import_folder` for imports

from ribo_EM import ribo_EM
from asite_predict import PredictAsite
import argparse
import pandas as pd

# Initialize abundance of all transcript to 1/#transcripts


# # read in the sequence
# cdsfasta = '/fast/work/groups/ag_ohler/dharnet_m/cortexomics/ext_data/gencode.vM12.pc_transcripts.fa'
# # transcript bam
# bam = 'pipeline/star_transcript/data/E13_ribo_1/E13_ribo_1.sort.bam'


#Note - this function needs a fasta file with teh following format - (coordinates 1 indexed)
#>ENSMUST00000070533.4|ENSMUSG00000051951.5|OTTMUSG00000026353.2|OTTMUST00000065166.1|Xkr4-001|Xkr4|3634|UTR5:1-150|CDS:151-2094|UTR3:2095-3634|
def transcript_initialize(cdsfasta):

    f_ref = open(cdsfasta)
    # transcript_TPMs = {}
    transcript_CDS_len = {}
    transcript_cdsstart = {}
    transcript_cdsend = {}
    transcript_gene = {}
    exonseq = {}
    for line in f_ref:
        if line.startswith('>'):
            line = line.lstrip('>').rstrip().rstrip('|')
            linearray = line.split('|')
            transID = linearray[0]
            transcript_gene[transID] = linearray[1]
            exonseq[transID] = ''
            # transcript_TPMs[transID] = 1
            for a in linearray:
                if a.startswith("CDS:"):
                    start, stop = a.lstrip("CDS:").split('-')
                    transcript_CDS_len[transID] = int(stop) - int(start) + 1
                    transcript_cdsstart[transID] = int(start)-1
                    transcript_cdsend[transID] = int(stop)-1
        else:
            exonseq[transID] += line.strip()
    # transNum = len(transcript_TPMs)
    # for key, value in transcript_TPMs.items():
        # transcript_TPMs[key] = 1000000.0/transNum
    f_ref.close()

    # return [transcript_TPMs, transcript_CDS_len, transcript_cdsstart, transcript_cdsend, exonseq]
    return [transcript_CDS_len, transcript_cdsstart, transcript_cdsend, exonseq, transcript_gene]


def make_bamDF(bam, mapq=-1, minRL=20, maxRL=35):
    # create a template/header from the input bam file
    inBam = pysam.AlignmentFile(bam, "rb")
    # read a bam file and extract info
    # cigar_to_exclude = set([1,2,3,4,5]) #set(['I','D','S','H'])
    cigar_to_exclude = set([1, 2, 3, 4, 5])  # set(['I','D','S','H'])
    i = 0
    readdf = []
    for read in inBam.fetch():
        i += 1
        # if(i==10000):break
        cigars = set([c[0] for c in read.cigartuples])
        if read.mapping_quality > mapq and \
                minRL <= read.query_length <= maxRL and \
                not cigars.intersection(cigar_to_exclude) and \
                not read.is_reverse:
                # The start is 0 indexed, the end is 1 indexed! Make them
                # both 0 indexed, so if I index a string with the trseq using the cols
                # I get the first and last bp of the read
            readdf.append((read.query_name, read.reference_name.split(
                '|')[0], read.reference_start, read.reference_end-1, read.query_length))

    readdf = pd.DataFrame.from_records(
        readdf, columns=['read_name', 'tr_id', 'start', 'end', 'read_length'])

    return readdf


def add_read_seqcols(uniquereadpos,trlengths,exonseq):
    bamtrlens = trlengths[uniquereadpos.tr_id].reset_index(drop=True)
    uniquereadpos = uniquereadpos.reset_index(
        drop=True)[uniquereadpos.end.add(1).reset_index(drop=True) != bamtrlens]
    uniquereadpos = uniquereadpos[uniquereadpos.start != 0]
    # get the edge sequences of our unique read pos
    edgeseq = [(exonseq[tr][st-1], exonseq[tr][st], exonseq[tr][e], exonseq[tr][e+1]) for
               i, tr, st, e in uniquereadpos.itertuples()]
    # concat these to the unique read df
    seqcols = ['nt_-1', 'nt_0', 'nt_n-1', 'nt_n']
    edgeseq = pd.DataFrame.from_records(edgeseq, columns=seqcols)
    uniquereadpos = pd.concat(
        [uniquereadpos.reset_index(drop=True), edgeseq], axis=1)
    return uniquereadpos

def add_read_cdscols(uniquereadpos,transcript_cdsstart,transcript_cdsend):
    cdsstarts = pd.Series(transcript_cdsstart)[
        uniquereadpos.tr_id].reset_index(drop=True)
    uniquereadpos['cdspos'] = (uniquereadpos.start.values - cdsstarts).values

    uniquereadpos['cdsendpos'] = (uniquereadpos.start.values - \
        (pd.Series(transcript_cdsend) -
         2)[uniquereadpos.tr_id].reset_index(drop=True)).values

    uniquereadpos['5_offset'] = uniquereadpos['cdspos'] % 3
    uniquereadpos['3_offset'] = (
        uniquereadpos['cdspos']+(uniquereadpos['end']-uniquereadpos['start']+1)-1) % 3

    return uniquereadpos

def add_seqinfo(bamdf, transcript_cdsstart, exonseq, trlengths):
    # add sequence
    uniquereadpos = bamdf[['tr_id', 'start', 'end']].drop_duplicates()
    uniquereadpos = add_read_seqcols(uniquereadpos,trlengths,exonseq)
    uniquereadpos = add_read_cdscols(uniquereadpos,transcript_cdsstart,transcript_cdsend)
    # merge in to the nonredundant read df
    bamdf = pd.merge(
        bamdf, uniquereadpos,
        left_on=['tr_id', 'start', 'end'],
        right_on=['tr_id', 'start', 'end']
    )
    return bamdf


def makePredTraining(bamdf):

    training = bamdf[
        (bamdf['cdspos'] <= 0) &
        (bamdf['cdspos'] >= -bamdf['read_length'])
    ]

    training['asite'] = 3 - training['cdspos']

    training = (training.query('asite >= 9').
                query('asite <= 18').
                query('asite >= (read_length / 2 - 1)')
                )
    training = training[['read_length', 'nt_-1', 'nt_0',
                         'nt_n-1', 'nt_n', '5_offset', '3_offset', 'asite']]
    return training

def get_coddf(trstouse, transcript_cdsstart, transcript_cdsend, exonseq, trlength ):
    coddfs = []
    # def tcode():
    for tr in trstouse:
        futrcodpos = list(reversed(range(transcript_cdsstart[tr]-3, -1, -3)))
        cdscodpos = list(range(transcript_cdsstart[tr], transcript_cdsend[tr], 3))
        tputrcodpos = list(range(transcript_cdsend[tr]+1, trlength[tr]-2, 3))
        codstarts = futrcodpos + cdscodpos + tputrcodpos
        codidx = list(reversed(range(-1, -len(futrcodpos)-1, -1)))
        codidx += list(range(0, len(cdscodpos)))
        codidx += list(range(len(cdscodpos), len(cdscodpos)+len(tputrcodpos)))
        cods = wrap(exonseq[tr][codstarts[0]:codstarts[-1]+3], 3)
        coddf = pd.DataFrame(zip(codstarts, codidx, cods))
        coddf.columns = ['start', 'codon_idx', 'codon']
        coddf['end'] = coddf['start']+3
        coddf['tr_id'] = tr
        coddfs.append(coddf)

    # %prun -l 10 tcode()
    coddf = pd.concat(coddfs,axis=0)
    #
    #
    return coddf


""
if __name__ == '__main__':

    sys.argv=['']
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input bam file",default = "pipeline/weinberg_yeast_riboAligned.out.sort.bam")
    parser.add_argument("-f", help="Fasta file", default="pipeline/yeast_transcript.ext.fa")
    parser.add_argument("-v" , default=False ,help="verbose EM ?", dest='verbose', action='store_true')
    parser.add_argument("-e" , default=False ,help="expression only ?", dest='expr_only', action='store_true')
    parser.add_argument("-o", help="ribotrans_test", default="ribotranstest")
    args = parser.parse_args()
    

    bam = args.i
    cdsfasta = args.f

    transcript_CDS_len, transcript_cdsstart, transcript_cdsend, exonseq, transcript_gene = transcript_initialize(
        cdsfasta)
    trlength = pd.Series(dict([(s, len(exonseq[s])) for s in exonseq]))

    bamdf = make_bamDF(bam)

    print('bam loaded')

    #count only reads overlapping the cds
    bamdf = (bamdf.
        merge(pd.Series(transcript_cdsstart,name='cdsstart'),left_on='tr_id',right_index=True).
        query('end > cdsstart').
        drop(['cdsstart'],axis=1).
        merge(pd.Series(transcript_cdsend,name='cdsend'),left_on='tr_id',right_index=True).
        query('start < cdsend').
        drop(['cdsend'],axis=1)
    )

    bamdf = add_read_cdscols(bamdf,transcript_cdsstart,transcript_cdsend)

    ###############################################################################
    # ## now calculate expression levels
    
    sampreads, transcript_TPMs, TPM_diff, transcript_readcount = ribo_EM(
        bamdf[['read_name', 'tr_id']],
        transcript_CDS_len,
        numloops=1,
        verbose=args.verbose
    )
    
    counts = sampreads.groupby('tr_id').size()
    counts.name='read_count'
    
    tr_expr = pd.DataFrame(pd.Series(transcript_TPMs).reset_index()).rename(columns={'index':'tr_id',0:'RPF_dens'}).merge(
        pd.DataFrame(pd.Series(transcript_CDS_len).reset_index()).rename(columns={'index':'tr_id',0:'cds_len'})).merge(
        pd.DataFrame(pd.Series(counts).reset_index()).rename(columns={'index':'tr_id',0:'read_count'})
    )
    
    exprfile = args.o+'.psites.csv'.replace('.csv','')+'.tr_expr.tsv'
    tr_expr.to_csv(exprfile,index=False,sep='\t')
    print(exprfile)


    sampreads = (sampreads[['read_name', 'tr_id']].
        merge( bamdf['read_name read_length tr_id start cdspos 5_offset'.split()], how='inner')
        )
    sampreads = sampreads.rename({'5_offset':'phase'},axis=1)

   

    # this gets us atg and sotp codon
    # tr = np.random.choice(list(transcript_cdsstart.keys()))
    # exonseq[tr][transcript_cdsstart[tr]:transcript_cdsstart[tr]+3]
    # exonseq[tr][transcript_cdsend[tr]+1:transcript_cdsend[tr]+1+3]

    def get_cdsocc_offsets(bamdf):

        def tally(edgereads,gcols):
            edgecounts = edgereads.groupby(['tr_id','cdspos','cdsendpos','read_length']).size()
            edgecounts = edgecounts.reset_index()
            edgecounts = edgecounts.rename({0:'n'},axis=1)
            return edgecounts
        #get reads gained as we move teh cds left
        gaincounts = tally(
            bamdf.query('(cdspos <=0) & ((-cdspos) <= (read_length-1))'),
            ['tr_id','cdspos','cdsendpos','read_length']
        )
        gaincounts['phase'] = gaincounts['cdspos']%3
        #the offset required to gain it
        gaincounts['offset']=-(gaincounts['cdspos']-gaincounts['phase'])
        #get reads lost as we move the cds left
        losscounts = tally(
            bamdf.query('((cdsendpos-2)<=0) &(  (-(cdsendpos-2))<=read_length-1 )'),
            ['tr_id','cdspos','cdsendpos','read_length']
        )
        losscounts['phase'] = losscounts['cdspos']%3
        #the offset required to lose it
        losscounts['cdsendpos'] += -3
        losscounts['offset']= -(losscounts['cdsendpos']-losscounts['phase'])

        #now make a data frame with gained and lost at each offset
        glcols = 'tr_id offset phase read_length'.split()
        gldf = gaincounts[glcols+['n']].merge(losscounts[glcols+['n']],on=glcols)
  
        #TODO - find a read that's at the stop codons location, verify it's cdsendpos
        #now get all combs of values (rl,phase,offset,tr) and make sure there's a value
        #for each of these
        cols = ['read_length','phase','offset','tr_id']
        uniquevallist = [list(np.sort(gldf[col].unique())) for col in cols]
        fullposdf = pd.DataFrame.from_records(it.product(*uniquevallist),columns=cols)
        fullposdf = fullposdf.query('offset>3')
        allposdf = fullposdf.merge(gldf,how='left')
        allposdf.n_x = allposdf.n_x.fillna(0)
        allposdf.n_y = allposdf.n_y.fillna(0)
        #column showing net effect of moving to that offset
        allposdf['net'] = allposdf.n_x - allposdf.n_y 
        #select trs for use in our survey
        # trs2use = allposdf.tr_id.unique() 
        # allposdf2 = allposdf[allposdf.tr_id.isin(trs2use)]
        allposdf = allposdf.drop(['n_x','n_y'],axis=1)
        #allposdf = allposdf.groupby(['read_length','phase','tr_id'])[['net']]
        allposdfgrp = allposdf.groupby(['read_length','phase','tr_id'])[['offset','net']]
        # g = next(x for x in allposdf2grp)
        #we want, not just the best offset, but the best offsets (given ties might exist)
        #for each transcript
        def votefun(df):
            cs = df.net.cumsum()
            ofs = df['offset'][cs == cs.max()].values
            return tuple(ofs)
        #
        gvotes = allposdfgrp.apply(lambda x:votefun(x)).explode()
        gvotes = pd.DataFrame(gvotes)
        gvotes = gvotes.rename({0:'offset'},axis=1)
        offsetvotes = gvotes.groupby(['read_length','phase'])['offset'].value_counts()
        offsetvotes.name = 'n_genes'
        #we now have a data frame with the number of genes indicating this offset
        #for each rl,phase,offset
        offsetvotes = offsetvotes.reset_index()
        #we then take the best, based on the offset compatible with the most genes
        bestoffsetvotes = (offsetvotes.groupby(['read_length','phase']).
            apply(lambda x: x.loc[x.n_genes.idxmax()]) )
        bestoffsetvotes = bestoffsetvotes.reset_index(drop=True)

        #more simply, we can just get this by combining all trs...        
        bestoffsetsum = allposdf.groupby(['read_length','offset','phase'])['net'].sum().groupby(['read_length','phase']).cumsum().groupby(['read_length','phase']).idxmax()
        pd.DataFrame(bestoffsetsum) 
        bestoffsetsum = pd.DataFrame(bestoffsetsum).reset_index()
        bestoffsetsum['bestoffset'] = bestoffsetsum['net'].str[1]
        bestoffsetsum = bestoffsetsum.drop('net',axis=1)
        #now print the best offsets

        return bestoffsetvotes,bestoffsetsum

    bestoffsetvotes,bestoffsetsum = get_cdsocc_offsets(bamdf)
    

    #saving work in progress
    import shelve
    filename='image.shelve.out'
    my_shelf = shelve.open(filename,'n') # 'n' for new
    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()

    # my_shelf = shelve.open(filename)
    # for key in my_shelf:
    #     globals()[key]=my_shelf[key]
    # my_shelf.close()


    # bestoffsetvotes = 
    # sampreads.merge(bestoffsetvotes[['read_length','phase','offset']])
    sampreads = sampreads.merge(bestoffsetvotes[['read_length','phase','offset']])
    sampreads = sampreads.assign(cdspos = lambda df:df.cdspos + df.offset)
    sampreads = sampreads.assign(start = lambda df:df.start + df.offset)
    sampreads = sampreads.drop('offset',axis=1)
    sampreads = sampreads.assign(codon_idx = lambda df: (df['cdspos'] - df['phase'])%3)
    
    gtpms = pd.concat([pd.Series(transcript_gene,name='gene'),pd.Series(dict(transcript_TPMs),name='TPM')],axis=1)
    TPMTHRESH = 0
    #get trs above threshold, but also the best for the gene
    trstouse = gtpms.query('TPM>@TPMTHRESH').sort_values('TPM',ascending=False).groupby('gene').head(1)['TPM'].index.to_series()
    # 
    coddf = get_coddf(trstouse, transcript_cdsstart, transcript_cdsend, exonseq, trlength)

    # sampreadssamp = sampreads.head(10000)
    sampreadssamp = sampreads
    # wide = (sampreadssamp.groupby(['tr_id','start','cdspos','read_length','phase']).
    #     size().
    #     reset_index().
    #     pivot_table(index=['tr_id','start','cdspos'],
    #         columns=['read_length','phase'])
    # )
    # wide.columns = [str(c[1])+'_'+str(c[2]) for c in wide.columns]
    sampreadssamp = sampreadssamp.assign(codon_idx=lambda x:(x.cdspos-x['phase'])/3)
    sumpsites = sampreadssamp.groupby(['tr_id','codon_idx']).size().reset_index().rename({0:'ribosome_count'},axis=1)
    sumpsites = coddf[['tr_id','codon_idx','codon']].merge(sumpsites[['tr_id','codon_idx','ribosome_count']],how='left')
    sumpsites.ribosome_count = sumpsites.ribosome_count.fillna(0) 
    
    sumpsites.to_csv(args.o+'.psites.csv',index=False)
    print(args.o+'.psites.csv')
    tpmdf = pd.Series(transcript_TPMs).reset_index().rename({'index':'tr_id',0:'TPM'},axis=1)
    tpmdf.to_csv(args.o+'.ribotpm.csv',index=False)
    print(args.o+'.ribotpm.csv')
    cdsdims = pd.concat([pd.Series(transcript_cdsstart),pd.Series(transcript_cdsend)+1,trlength],axis=1) 
    cdsdims.columns=['aug','stop','length']
    cdsdims.to_csv(args.o+'.cdsdims.csv',index=False)
    print(args.o+'.cdsdims.csv')

    tr=sumpsites.tr_id[0]
    NTOKS=512
    NFLANK=5
#    codons_available = trcodons.n_cods+(2*NFLANK)


    psitetrs=sumpsites.tr_id.unique()
    trcodons = ((cdsdims.stop-cdsdims.aug ) / 3)[psitetrs]
    #include flanks in this number
    trcodons = trcodons + (2*NFLANK)
    trcodons.name = 'n_cods'
    trcodons.index.name = 'tr_id'
    excesscodons = ((trcodons)-NTOKS).clip(0)
    # trcodons = trcodons.reset_index()
    excesscodons[tr]
    trcodons[tr]



    # firsthalfend = cdsdims.aug + np.ceil((trcodons/2)).clip(upper=256)
    firsthalfend = np.ceil((trcodons/2)).clip(upper=NTOKS/2)-NFLANK#1based
    firsthalfend[tr]
    sechalfstart = (trcodons-NFLANK) - np.floor((trcodons/2)).clip(upper=NTOKS/2)
    sechalfstart[tr]

    #these two are the right size, it seems
    firsthalfend[tr] - (0-NFLANK)
    (trcodons[tr]-NFLANK)-sechalfstart[tr]
    (sechalfstart - firsthalfend)==excesscodons

    #so this defines the middle region, in 0,1 index
    middlestart=firsthalfend
    middleend=sechalfstart

    sumpsites2 = sumpsites

    sumpsites2 = sumpsites2[sumpsites2.codon_idx<(trcodons[sumpsites2.tr_id]-NFLANK).values]
    sumpsites2 = sumpsites2[sumpsites2.codon_idx>= -NFLANK]
    inmiddle = (firsthalfend[sumpsites2.tr_id].values<=sumpsites2.codon_idx) & (sumpsites2.codon_idx < sechalfstart[sumpsites2.tr_id].values)
    sumpsites2 = sumpsites2[~inmiddle]

    assert (sumpsites2.groupby('tr_id').size()[trcodons.index] == trcodons.clip(upper=NTOKS)).all()

    sumpsites = sumpsites2

    ################################################################################
    ########Now parse into tensors
    ################################################################################
    


    sumpsites2.groupby('tr_id').size().max()

    sumpsites.merge(trcodons,).query('(codon_idx+1) <= (n_cods - NFLANK)')
    sumpsites[sumpsites.tr_id==tr]


    tr='YPR204C-A' 
    trcodons[tr]
    #this looks right, so that e.g. in a tr with 160 codons, codon_idx==159 is the last cododing codon, and 160 is the stop
    coddf.query('codon_idx >= -3').query('tr_id=="YPR204C-A"').merge(trcodons).query('codon_idx < n_cods+3')








    # # this gets us atg and sotp codon
    # tr = np.random.choice(list(transcript_cdsstart.keys()))
    # exonseq[tr][transcript_cdsstart[tr]:transcript_cdsstart[tr]+3]
    # exonseq[tr][transcript_cdsend[tr]+1:transcript_cdsend[tr]+1+3]
    # exonseq[tr][transcript_cdsstart[tr]:transcript_cdsend[tr]+1]
    # exonseq[tr][transcript_cdsstart[tr]:transcript_cdsend[tr]+1+3]
