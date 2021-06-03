import argparse
import pathlib
import pysam  # type: ignore[import]
import numpy as np
import pandas as pd
import itertools as it
from pathlib import Path
from textwrap import wrap
from ribo_EM import ribo_EM


class Cdsannotation:
    def __init__(self, cdsfasta: pathlib.Path):
        f_ref = open(cdsfasta)
        self.transcript_CDS_len = {}
        self.transcript_cdsstart = {}
        self.transcript_cdsend = {}
        self.transcript_gene = {}
        self.exonseq = {}
        for line in f_ref:
            if line.startswith('>'):
                line = line.lstrip('>').rstrip().rstrip('|')
                linearray = line.split('|')
                transID = linearray[0]
                self.transcript_gene[transID] = linearray[1]
                self.exonseq[transID] = ''
                for a in linearray:
                    if a.startswith("CDS:"):
                        start, stop = a.lstrip("CDS:").split('-')
                        cdslen = int(stop) - int(start) + 1
                        self.transcript_CDS_len[transID] = cdslen
                        self.transcript_cdsstart[transID] = int(start) - 1
                        self.transcript_cdsend[transID] = int(stop) - 1
            else:
                self.exonseq[transID] += line.strip()
        self.trlength: pd.Series = pd.Series(
            dict([(s, len(self.exonseq[s])) for s in self.exonseq]))

        f_ref.close()
        #


def make_bamDF(
    bam: Path, mapq: int = -1,
    minRL: int = 20, maxRL: int = 35
) -> pd.DataFrame:

    # create a template/header from the input bam file
    inBam = pysam.AlignmentFile(bam, "rb")
    # read a bam file and extract infoG
    # cigar_to_exclude = set([1,2,3,4,5]) #set(['I','D','S','H'])
    cigar_to_exclude = set([1, 2, 3, 4, 5])  # set(['I','D','S','H'])
    i = 0
    readdf = []
    for read in inBam.fetch():
        i += 1
        # if(i == 50000):
            # break
        cigars = set([c[0] for c in read.cigartuples])
        if read.mapping_quality > mapq and \
                minRL <= read.query_length <= maxRL and \
                not cigars.intersection(cigar_to_exclude) and \
                not read.is_reverse:
            # The start is 0 indexed, the end is 1 indexed! Make them
            # both 0 indexed, so index a string with the trseq using the cols
            # I get the first and last bp of the read
            readdf.append((read.query_name, read.reference_name.split(
                '|')[0], read.reference_start,
                read.reference_end - 1, read.query_length))

    outdf: pd.DataFrame = pd.DataFrame.from_records(
        readdf, columns=['read_name', 'tr_id', 'start', 'end', 'read_length'])

    return outdf

###############################################################################
########
###############################################################################


def add_read_seqcols(uniquereadpos, cdsanno):
    bamtrlens = cdsanno.trlengths[uniquereadpos.tr_id].reset_index(drop=True)
    ind = uniquereadpos.end.add(1).reset_index(drop=True) != bamtrlens
    uniquereadpos = uniquereadpos.reset_index(
        drop=True)[ind]
    uniquereadpos = uniquereadpos[uniquereadpos.start != 0]
    # get the edge sequences of our unique read pos
    edgeseq = [(cdsanno.exonseq[tr][st - 1], cdsanno.exonseq[tr][st],
                cdsanno.exonseq[tr][e], cdsanno.exonseq[tr][e + 1]) for
               i, tr, st, e in uniquereadpos.itertuples()]
    # concat these to the unique read df
    seqcols = ['nt_-1', 'nt_0', 'nt_n-1', 'nt_n']
    edgeseq = pd.DataFrame.from_records(edgeseq, columns=seqcols)
    uniquereadpos = pd.concat(
        [uniquereadpos.reset_index(drop=True), edgeseq], axis=1)
    return uniquereadpos


def add_read_cdscols(upos, cdsanno):
    cdsstarts = pd.Series(cdsanno.transcript_cdsstart)[
        upos.tr_id].reset_index(drop=True)
    upos['cdspos'] = (upos.start.values - cdsstarts).values
    # position relative to end
    cdsend = (pd.Series(cdsanno.transcript_cdsend) - 2)[upos.tr_id]
    cdsend = cdsend.reset_index(drop=True)
    upos['cdsendpos'] = (upos.start.values - cdsend).values
    # offsets
    upos['5_offset'] = upos['cdspos'] % 3
    upos['3_offset'] = (
        upos['cdspos'] + (upos['end'] - upos['start'] + 1) - 1) % 3

    return upos


def add_seqinfo(bamdf, cdsanno):
    # add sequence
    uniquereadpos = bamdf[['tr_id', 'start', 'end']].drop_duplicates()
    uniquereadpos = add_read_seqcols(uniquereadpos, cdsanno)
    uniquereadpos = add_read_cdscols(
        uniquereadpos, cdsanno.transcript_cdsstart, cdsanno.transcript_cdsend)
    # merge in to the nonredundant read df
    bamdf = pd.merge(
        bamdf, uniquereadpos,
        left_on=['tr_id', 'start', 'end'],
        right_on=['tr_id', 'start', 'end']
    )
    return bamdf


def makePredTraining(bamdf):
    nottooleft = (bamdf['cdspos'] >= -bamdf['read_length'])
    nottooright = (bamdf['cdspos'] <= 0)
    training = bamdf[nottooright & nottooleft]

    training['asite'] = 3 - training['cdspos']

    training = (training.query('asite >= 9').
                query('asite <= 18').
                query('asite >= (read_length / 2 - 1)')
                )
    training = training[['read_length', 'nt_-1', 'nt_0',
                         'nt_n-1', 'nt_n', '5_offset', '3_offset', 'asite']]
    return training


def get_coddf(trstouse, cdsanno):
    coddfs = []
    # def tcode():
    for tr in trstouse:
        start = cdsanno.transcript_cdsstart[tr]
        ends = cdsanno.transcript_cdsend[tr]
        futrcodpos = range(start - 3, -1, -3)
        futrcodpos = list(reversed(futrcodpos))
        cdscodpos = list(
            range(start, ends, 3))
        tputrcodpos = list(range(ends + 1, cdsanno.trlength[tr] - 2, 3))
        codstarts = futrcodpos + cdscodpos + tputrcodpos
        codidx = list(reversed(range(-1, -len(futrcodpos) - 1, -1)))
        codidx += list(range(0, len(cdscodpos)))
        codidx += list(range(
            len(cdscodpos),
            len(cdscodpos) + len(tputrcodpos)
        ))
        cods = wrap(cdsanno.exonseq[tr][codstarts[0]:codstarts[-1] + 3], 3)
        coddf = pd.DataFrame(zip(codstarts, codidx, cods))
        coddf.columns = ['start', 'codon_idx', 'codon']
        coddf['end'] = coddf['start'] + 3
        coddf['tr_id'] = tr
        coddfs.append(coddf)

    # %prun -l 10 tcode()
    coddf = pd.concat(coddfs, axis=0)
    #
    #
    return coddf


""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="Input bam file",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default='../../cortexomics/pipeline/star_transcript/data/E13_ribo_1/'\
        'E13_ribo_1.sort.bam'
    )
    parser.add_argument(
        "-f", help="Fasta file",
        default='../../cortexomics/ext_data/gencode.vM12.pc_transcripts.fa'
    )
    parser.add_argument(
        "-l",
        help="File of read length, optionally phase, offset",
        default="../../cortexomics/ext_data/offsets_manual.tsv"
    )
    parser.add_argument("-v", default=False, help="verbose EM ?",
                        dest='verbose', action='store_true')
    parser.add_argument("-e", default=False, help="expression only ?",
                        dest='expr_only', action='store_true')
    parser.add_argument("-o", help="cortexomicstest",
                        default="cortexomicstest"
                        )
    args = parser.parse_args()

    bam: Path = Path(args.i)
    cdsfasta: Path = Path(args.f)

    cdsanno = Cdsannotation(cdsfasta)

    bamdf = make_bamDF(bam)
    assert (bamdf.read_length == 31).any()

    print('bam loaded')

    # count only reads overlapping the cds
    startseries = pd.Series(cdsanno.transcript_cdsstart, name='cdsstart')
    endseries = pd.Series(cdsanno.transcript_cdsend, name='cdsend')
    bamdf = (bamdf.
             merge(startseries, left_on='tr_id', right_index=True).
             query('end > cdsstart').
             drop(['cdsstart'], axis=1).
             merge(endseries, left_on='tr_id', right_index=True).
             query('start < cdsend').
             drop(['cdsend'], axis=1)
             )

    bamdf = add_read_cdscols(bamdf, cdsanno)
    # ## now calculate expression levels

    sampreads, transcript_TPMs, TPM_diff, transcript_readcount = ribo_EM(
        bamdf[['read_name', 'tr_id']],
        cdsanno.transcript_CDS_len,
        numloops=50,
        verbose=args.verbose
    )

    counts = sampreads.groupby('tr_id').size()
    counts.name = 'read_count'

    tpmseries = (
        pd.DataFrame(pd.Series(transcript_TPMs).reset_index()).
        rename(columns={'index': 'tr_id', 0: 'RPF_dens'}))
    lenseries = (
        pd.DataFrame(pd.Series(cdsanno.transcript_CDS_len).reset_index()).
        rename(columns={'index': 'tr_id', 0: 'cds_len'}))
    countseries = (
        pd.DataFrame(pd.Series(counts).reset_index()).
        rename(columns={'index': 'tr_id', 0: 'read_count'}))
    tr_expr = tpmseries.merge(lenseries).merge(countseries)

    exprfile = args.o + '.psites.csv'.replace('.csv', '') + '.tr_expr.tsv'
    tr_expr.to_csv(exprfile, index=False, sep='\t')
    print(exprfile)

    cols = 'read_name read_length tr_id start cdspos 5_offset'.split()
    sampreads = (sampreads[['read_name', 'tr_id']].
                 merge(
                     bamdf[cols], how='inner')
                 )
    sampreads = sampreads.rename({'5_offset': 'phase'}, axis=1)

    def get_cdsocc_offsets(bamdf):

        def tally(edgereads, gcols):
            edgecounts = edgereads.groupby(
                ['tr_id', 'cdspos', 'cdsendpos', 'read_length']).size()
            edgecounts = edgecounts.reset_index()
            edgecounts = edgecounts.rename({0: 'n'}, axis=1)
            return edgecounts
        # get reads gained as we move teh cds left
        gaincounts = tally(
            bamdf.query('(cdspos <=0) & ((-cdspos) <= (read_length-1))'),
            ['tr_id', 'cdspos', 'cdsendpos', 'read_length']
        )
        gaincounts['phase'] = gaincounts['cdspos'] % 3
        # the offset required to gain it
        gaincounts['offset'] = -(gaincounts['cdspos'] - gaincounts['phase'])
        # get reads lost as we move the cds left
        losscounts = tally(
            bamdf.query(
                '((cdsendpos-2)<=0) &(  (-(cdsendpos-2))<=read_length-1 )'),
            ['tr_id', 'cdspos', 'cdsendpos', 'read_length']
        )
        losscounts['phase'] = losscounts['cdspos'] % 3
        # the offset required to lose it
        losscounts['cdsendpos'] += -3
        losscounts['offset'] = -(losscounts['cdsendpos'] - losscounts['phase'])

        # now make a data frame with gained and lost at each offset
        glcols = 'tr_id offset phase read_length'.split()
        gldf = gaincounts[glcols + ['n']
                          ].merge(losscounts[glcols + ['n']], on=glcols)

        # now get all combs of values (rl,phase,offset,tr)
        # and make sure there's a value
        # for each of these
        cols = ['read_length', 'phase', 'offset', 'tr_id']
        uniquevallist = [list(np.sort(gldf[col].unique())) for col in cols]
        fullposdf = pd.DataFrame.from_records(
            it.product(*uniquevallist), columns=cols)
        fullposdf = fullposdf.query('offset>3')
        allposdf = fullposdf.merge(gldf, how='left')
        allposdf.n_x = allposdf.n_x.fillna(0)
        allposdf.n_y = allposdf.n_y.fillna(0)
        # column showing net effect of moving to that offset
        allposdf['net'] = allposdf.n_x - allposdf.n_y
        # select trs for use in our survey
        # trs2use = allposdf.tr_id.unique()
        # allposdf2 = allposdf[allposdf.tr_id.isin(trs2use)]
        allposdf = allposdf.drop(['n_x', 'n_y'], axis=1)
        # allposdf = allposdf.groupby(['read_length','phase','tr_id'])[['net']]
        allposdfgrp = allposdf.groupby(['read_length', 'phase', 'tr_id'])[
            ['offset', 'net']]
        # g = next(x for x in allposdf2grp)
        # we want, not just the best offset, but the best offsets
        # (given ties might exist)
        # for each transcript

        def votefun(df):
            cs = df.net.cumsum()
            ofs = df['offset'][cs == cs.max()].values
            return tuple(ofs)
        #
        gvotes = allposdfgrp.apply(lambda x: votefun(x)).explode()
        gvotes = pd.DataFrame(gvotes)
        gvotes = gvotes.rename({0: 'offset'}, axis=1)
        offsetvotes = gvotes.groupby(['read_length', 'phase'])[
            'offset'].value_counts()
        offsetvotes.name = 'n_genes'
        # we now have a data frame with the n of trs indicating this offset
        # for each rl,phase,offset
        offsetvotes = offsetvotes.reset_index()
        # take the best, based on offset compatible with the most trs
        bestoffsetvotes = (offsetvotes.groupby(['read_length', 'phase']).
                           apply(lambda x: x.loc[x.n_genes.idxmax()]))
        bestoffsetvotes = bestoffsetvotes.reset_index(drop=True)

        # more simply, we can just get this by combining all trs...
        bestoffsetsum = (
            allposdf.
            groupby(['read_length', 'offset', 'phase'])['net'].
            sum().
            groupby(['read_length', 'phase']).
            cumsum().
            groupby(['read_length', 'phase']).
            idxmax()
        )
        bestoffsetsum = pd.DataFrame(bestoffsetsum).reset_index()
        bestoffsetsum['bestoffset'] = bestoffsetsum['net'].str[1]
        bestoffsetsum = bestoffsetsum.drop('net', axis=1)
        # now print the best offsets

        return bestoffsetvotes, bestoffsetsum

    if args.l is not None:
        bestoffsetvotes, bestoffsetsum = get_cdsocc_offsets(bamdf)
    else:
        bestoffsetvotes = pd.read_csv(args.l, sep='\t')
        if 'phase' not in bestoffsetvotes.columns:
            bestoffsetvotes = pd.concat([
                bestoffsetvotes.assign(phase=0),
                bestoffsetvotes.assign(phase=1),
                bestoffsetvotes.assign(phase=2)
            ])
        bestoffsetvotes = bestoffsetvotes.rename(
            {'length': 'read_length'}, axis=1)

        bestoffsetvotes = bestoffsetvotes.rename(
            {'cutoff': 'offset'}, axis=1)

    sampreads = sampreads.merge(
        bestoffsetvotes[['read_length', 'phase', 'offset']])

    sampreads = sampreads.assign(cdspos=lambda df: df.cdspos + df.offset)
    sampreads = sampreads.assign(start=lambda df: df.start + df.offset)
    sampreads = sampreads.drop('offset', axis=1)
    sampreads = sampreads.assign(
        codon_idx=lambda df: (df['cdspos'] - df['phase']) % 3)

    gtpms = pd.concat(
        [
            pd.Series(cdsanno.transcript_gene, name='gene'),
            pd.Series(dict(transcript_TPMs), name='TPM')
        ],
        axis=1
    )

    TPMTHRESH = 0
    # get trs above threshold, but also the best for the gene
    trstouse = (
        gtpms.query('TPM>@TPMTHRESH').
        sort_values('TPM', ascending=False).
        groupby('gene').
        head(1)['TPM'].
        index.
        to_series()
    )
    #
    coddf = get_coddf(trstouse, cdsanno)

    sampreadssamp = sampreads

    sampreadssamp = sampreadssamp.assign(
        codon_idx=lambda x: (x.cdspos - x['phase']) / 3
    )
    sumpsites = sampreadssamp.groupby(['tr_id', 'codon_idx']).size(
    ).reset_index().rename({0: 'ribosome_count'}, axis=1)
    sumpsites = coddf[['tr_id', 'codon_idx', 'codon']].merge(
        sumpsites[['tr_id', 'codon_idx', 'ribosome_count']], how='left')
    sumpsites.ribosome_count = sumpsites.ribosome_count.fillna(0)

    sumpsites.to_csv(args.o + '.all.psites.csv', index=False)
    print(args.o + '.psites.csv')
    tpmdf = pd.Series(transcript_TPMs, name='TPM').reset_index(name='tr_id')
    tpmdf.to_csv(f'{args.o}.ribotpm.csv', index=False)

    print(f'{args.o}.ribotpm.csv')
    cdsdims: pd.DataFrame = pd.DataFrame(
        pd.concat(
            [
                pd.Series(cdsanno.transcript_cdsstart),
                pd.Series(cdsanno.transcript_cdsend) + 1,
                cdsanno.trlength],
            axis=1)
    )
    cdsdims.columns = ['aug', 'stop', 'length']

    tr = sumpsites.tr_id[0]
    NTOKS = 512
    NFLANK = 5
#    codons_available = trcodons.n_cods+(2*NFLANK)

    # testr = 'ENSMUST00000184432.2'

    def trim_middle(sumpsites, cdsdims, NFLANK):
        psitetrs = sumpsites.tr_id.unique()
        trcodons = ((cdsdims.stop - cdsdims.aug) / 3)[psitetrs]
        # include flanks in this number
        trcodons = trcodons + (2 * NFLANK)
        trcodons.name = 'n_cods'
        trcodons.index.name = 'tr_id'
        # trcodons = trcodons.reset_index().rename({'index':'tr_id'},axis=1)
        # excesscodons[tr]
        # trcodons[tr]

        # firsthalfend = cdsdims.aug + np.ceil((trcodons/2)).clip(upper=256)
        firsthalfend = np.ceil(
            (trcodons / 2)).clip(upper=NTOKS / 2) - NFLANK  # 1based
        # firsthalfend[tr]
        sechalfstart = (trcodons - NFLANK) -  \
            np.floor((trcodons / 2)).clip(upper=NTOKS / 2)
        # sechalfstart[tr]

        # these two are the right size, it seems
        # firsthalfend[tr]  -  (0 - NFLANK)
        # (trcodons[tr] - NFLANK) - sechalfstart[tr]
        # (sechalfstart  -  firsthalfend)==excesscodons

        sp2 = sumpsites

        sp2 = sp2[sp2.codon_idx < (trcodons[sp2.tr_id] - NFLANK).values]

        sp2 = sp2[sp2.codon_idx >= - NFLANK]
        inmiddle = (firsthalfend[sp2.tr_id].values <= sp2.codon_idx) & (
            sp2.codon_idx < sechalfstart[sp2.tr_id].values)
        sp2 = sp2[~inmiddle]

        # this assertion only works where
        # everything has big enough UTRs to include the flanks
        # assert (sp2.groupby('tr_id').size()[
        # trcodons.index] == trcodons.clip(upper=NTOKS)).all()

        return sp2

    sumpsites = trim_middle(sumpsites, cdsdims, NFLANK)

    sumpsites.to_csv(f'{args.o}.sumpsites.csv', index=False)

    print(f'{args.o}.sumpsites.csv')

    cdsdims = cdsdims.reset_index().rename({'index': 'tr_id'}, axis=1)
    cdsdims.to_csv(args.o + '.cdsdims.csv', index=False)
    print(f'{args.o}.cdsdims.csv')

# # this gets us atg and sotp codon
# tr = np.random.choice(list(cdsanno.transcript_cdsstart.keys()))
# cdsanno.exonseq[tr][cdsanno.transcript_cdsstart[tr]:cdsanno.transcript_cdsstart[tr]+3]
# cdsanno.exonseq[tr][cdsanno.transcript_cdsend[tr]+1:cdsanno.transcript_cdsend[tr]+1+3]
# cdsanno.exonseq[tr][cdsanno.transcript_cdsstart[tr]:cdsanno.transcript_cdsend[tr]+1]
# cdsanno.exonseq[tr][cdsanno.transcript_cdsstart[tr]:cdsanno.transcript_cdsend[tr]+1+3]
