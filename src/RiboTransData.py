import sys
import torch
import numpy as np
import random
import copy
import argparse
import torch.nn as nn
import pandas as pd

from collections import OrderedDict
from ixnosdata import Ixnosdata
import torch.utils.data as data
from train_ixmodel import Net
from pathlib import Path

ten = torch.Tensor
NTOKS = 512
RIBODENS_THRESH = 0.5


def trim_middle(sumpsites, cdsdims, NFLANK, NTOKS):
    """Take in a data frame of codons and cds dimensions and trim out
    the middle of each one so that we get NFLANK non coding codons on either
    side and a total of NTOKS codons in the middle"""
    psitetrs = sumpsites.tr_id.unique()
    assert psitetrs[0] in cdsdims.index
    trcodons = ((cdsdims.stop - cdsdims.aug) / 3)[psitetrs]
    # include flanks in this number
    trcodons = trcodons + (2 * NFLANK)
    trcodons.name = 'n_cods'
    trcodons.index.name = 'tr_id'
    firsthalfend = np.ceil(
        (trcodons / 2)).clip(upper=NTOKS / 2) - NFLANK  # 1based
    # firsthalfend[tr]
    sechalfstart = (trcodons - NFLANK) -  \
        np.floor((trcodons / 2)).clip(upper=NTOKS / 2)
    sp2 = sumpsites
    sp2 = sp2[sp2.codon_idx < (trcodons[sp2.tr_id] - NFLANK).values]
    sp2 = sp2[sp2.codon_idx >= - NFLANK]
    inmiddle = (firsthalfend[sp2.tr_id].values <= sp2.codon_idx) & (
        sp2.codon_idx < sechalfstart[sp2.tr_id].values)
    sp2 = sp2[~inmiddle]
    return sp2


class RiboTransData(data.Dataset):
    CODON_TYPES = ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG',
                   'ATT', 'ATC', 'ATA', 'ATG', 'GTT', 'GTC', 'GTA',
                   'GTG', 'TCT', 'TCC', 'TCA', 'TCG', 'CCT', 'CCC', 'CCA',
                   'CCG', 'ACT', 'ACC', 'ACA', 'ACG', 'GCT', 'GCC',
                   'GCA', 'GCG', 'TAT', 'TAC', 'CAT', 'CAC', 'CAA', 'CAG',
                   'AAT', 'AAC', 'AAA', 'AAG', 'GAT', 'GAC', 'GAA',
                   'GAG', 'TGT', 'TGC', 'TGG', 'CGT', 'CGC', 'CGA', 'CGG',
                   'AGT', 'AGC', 'AGA', 'AGG', 'GGT', 'GGC', 'GGA',
                   'GGG', 'TAA', 'TAG', 'TGA']

    genetic_code = {'TTT': 'F', 'TCT': 'S', 'TAT': 'Y', 'TGT': 'C', 'TTC': 'F',
                    'TCC': 'S', 'TAC': 'Y', 'TGC': 'C',
                    'TTA': 'L', 'TCA': 'S', 'TAA': '*', 'TGA': '*', 'TTG': 'L',
                    'TCG': 'S', 'TAG': '*', 'TGG': 'W',
                    'CTT': 'L', 'CCT': 'P', 'CAT': 'H', 'CGT': 'R', 'CTC': 'L',
                    'CCC': 'P', 'CAC': 'H', 'CGC': 'R',
                    'CTA': 'L', 'CCA': 'P', 'CAA': 'Q', 'CGA': 'R', 'CTG': 'L',
                    'CCG': 'P', 'CAG': 'Q', 'CGG': 'R',
                    'ATT': 'I', 'ACT': 'T', 'AAT': 'N', 'AGT': 'S', 'ATC': 'I',
                    'ACC': 'T', 'AAC': 'N', 'AGC': 'S',
                    'ATA': 'I', 'ACA': 'T', 'AAA': 'K', 'AGA': 'R', 'ATG': 'M',
                    'ACG': 'T', 'AAG': 'K', 'AGG': 'R',
                    'GTT': 'V', 'GCT': 'A', 'GAT': 'D', 'GGT': 'G', 'GTC': 'V',
                    'GCC': 'A', 'GAC': 'D', 'GGC': 'G',
                    'GTA': 'V', 'GCA': 'A', 'GAA': 'E', 'GGA': 'G', 'GTG': 'V',
                    'GCG': 'A', 'GAG': 'E', 'GGG': 'G'}

    codonnum = pd.Series(range(0, len(CODON_TYPES)), index=CODON_TYPES) + 1
    basenum = pd.Series(range(0, 4), index=['A', 'C', 'G', 'T']) + 1

    # this creates an rdata object from a dataframe of ribosome counts,codons
    # and trims it to matcht he given token size, with a specified number of
    # codons either side of the cds
    def __init__(self, codcountfile, cdsdims_file,
                 NTOKS, RIBODENS_THRESH, rpad=5):

        psitecovtbl = pd.read_csv(
            codcountfile
        )
        assert psitecovtbl.shape[0] > 1000
        cdsdims = pd.read_csv(cdsdims_file)
        cdsdims = cdsdims.set_index('tr_id')
        psitecovtbl = trim_middle(psitecovtbl, cdsdims, RPAD, NTOKS)

        #
        assert psitecovtbl.codon.isin(self.codonnum.index).all()

        # now filter by ribosome density
        # (say by 0.5 - gets us top 80% of weinberg data)
        ribodens = psitecovtbl.groupby('tr_id').ribosome_count.mean()
        lowdensgenes = ribodens.index[ribodens < RIBODENS_THRESH].values
        psitecovtbl = psitecovtbl[~psitecovtbl.tr_id.isin(
            lowdensgenes)]
        ribodens = ribodens[~ribodens.index.isin(lowdensgenes)]
        #
        # self.lowdensgenes = lowdensgenes
        self.ugenes = psitecovtbl.tr_id.unique()

        n_genes = self.ugenes.shape[0]
        print('using ' + str(n_genes) + ' genes')

        # index of the gene position
        self.gene2num = pd.Series(
            range(0, len(self.ugenes)), index=self.ugenes)
        # index of the codon position
        codindex = psitecovtbl.groupby('tr_id').codon_idx.rank() - 1
        # ribosignal
        self.ribosignal = torch.sparse.FloatTensor(
            torch.LongTensor([
                self.gene2num[psitecovtbl.tr_id].values,
                codindex
            ]),
            torch.FloatTensor(psitecovtbl.ribosome_count.values),
            torch.Size([n_genes, NTOKS])).to_dense()
        assert self.ribosignal.shape == torch.Size([n_genes, NTOKS])

        # codons
        self.data = OrderedDict()
        self.data['codons'] = torch.sparse.FloatTensor(
            torch.LongTensor([
                self.gene2num[psitecovtbl.tr_id].values,
                codindex
            ]),
            torch.LongTensor(self.codonnum[psitecovtbl.codon].values),
            torch.Size([n_genes, NTOKS])).to_dense()

        self.data['codons'] = nn.functional.one_hot(
            self.data['codons']).transpose(2, 1).float()
        assert self.data['codons'].shape == torch.Size(
            [n_genes, self.codonnum.shape[0] + 1, NTOKS])
        cdslens = (self.data['codons'][:, 0, :] != 1).sum(1)
        frac_filled = (cdslens / 512.0)

        self.n_toks = NTOKS
        gmeans = self.ribosignal.mean(axis=1)
        self.offset = (gmeans / frac_filled).reshape([len(self.ugenes), 1])
        self.offset = self.offset.reshape([-1, 1, 1])

        self.data['seqfeats'] = []
        for i in range(1, 4):
            nucs = psitecovtbl.codon.str.split('', expand=True)[i]
            numseq = self.basenum[nucs.values].values
            denseseqfeats = torch.sparse.FloatTensor(
                torch.LongTensor([
                    self.gene2num[psitecovtbl.tr_id].values,
                    codindex
                ]),
                torch.LongTensor(numseq),
                torch.Size([n_genes, NTOKS])).to_dense()
            self.data['seqfeats'].append(nn.functional.one_hot(denseseqfeats))
        self.data['seqfeats'] = torch.cat(
            self.data['seqfeats'], axis=2).transpose(2, 1).float()

        # non_null_mask = riboob.data['codons'][:,0,:]!=1

        self.device = 'cpu'
        self.usedata = list(self.data.keys())
        self.batch_size = 2

        self.gene2num
        self.ugenes
        self.ribosignal

        # lpad
        lpads = psitecovtbl.groupby('tr_id').codon_idx.min().value_counts()
        lpad = list(lpads.index)
        assert len(lpad) == 1
        lpad = lpad[0]
        self.lpad = lpad
        # rpad
        self.rpad = rpad

        self.batchshuffle = True

        self.inputfile = codcountfile
        self.inputcdsfile = cdsdims_file

    #extract a subset of genes from an rdata object
    def subset(self, idx):
        # idx=idx[idx.index[0]]
        srdata = copy.deepcopy(self)
        srdata.gene2num = srdata.gene2num[idx]
        # NTOKS-rtensor[srdata.gene2num.values][:,0,:].sum(axis=1)
        for nm, rtensor in srdata.data.items():
            srdata.data[nm] = rtensor[srdata.gene2num.values]
        srdata.ribosignal = srdata.ribosignal[srdata.gene2num.values]
        srdata.ribosignal[:, 0:10]
        srdata.offset = srdata.offset[srdata.gene2num.values]
        srdata.gene2num = pd.Series(
            range(0, len(srdata.gene2num)), index=srdata.gene2num.index)
        srdata.ugenes = pd.Series(srdata.gene2num.index)
        return srdata

    #add a tensor to the data slot of the rdata object
    def add_tensor(self, esmgenes, esmtensor, tensname):
        nameov = pd.Series(esmgenes).isin(self.gene2num.index)
        intgenes = esmgenes[nameov]
        esmtensor = esmtensor[nameov]
        subset = self.subset(intgenes)
        subset.data[tensname] = esmtensor

    # batchinds=torch.randperm(len(self))[0:10]
    def get_batch(self, batchinds):
        device = self.device
        # genes = self.ugenes[batchinds]
        # concatenate various codon level values
        bdata = [d[batchinds]
                 for k, d in self.data.items() if k in self.usedata]
        # bdata = [d[batchinds] for d in self.data.values()]
        bdata = torch.cat(bdata, axis=1)
        target = self.ribosignal[batchinds]
        offset = self.offset[batchinds]
        # gns = self.gene2num.index[batchinds]
        return (bdata.to(device), offset.to(device)), target.to(device)

    def orfclip(self, bdata, lclip=0, rclip=0):
        """
        Args:
            bdata: Tensor, shape [batch, feats, pos]
            lclip: int
            rclip: int
        """
        # this function outputs a logical tensor that designates
        # positions so many places outside of ORF. lclip -1 gives you
        # 1 codn upstream of start. rclip -1 clips 1 position off the
        assert rclip <= 0
        rclip -= self.rpad
        assert lclip >= self.lpad
        lclip -= self.lpad
        ltens = bdata[:, 0, :] != 1
        # use pandas to get max pos per gn
        inddf = pd.DataFrame(ltens.nonzero().numpy())
        inddf.columns = ['gn', 'pos']
        maxpos = inddf.groupby('gn').pos.max().values
        # now assign 0 or 1 to the tensor
        for g in range(bdata.shape[0]):
            ltens[g] = 0
            rind = (1 + maxpos[g] + rclip)
            if lclip <= rind:
                ltens[g, lclip:rind] = 1
        return ltens

    # splits our data object for val/test.
    def split_data(self, valsize=500, tsize=500):
        n_genes = len(self)

        allinds = np.array(random.sample(range(0, n_genes), k=n_genes))
        tvsize = valsize + tsize
        traininds = self.gene2num.index[allinds[0:(len(allinds) - tvsize)]]
        testinds = self.gene2num.index[allinds[(
            len(allinds) - tvsize):(len(allinds) - valsize)]]
        valinds = self.gene2num.index[allinds[(len(allinds) - valsize):]]

        train_data = self.subset(traininds)
        test_data = self.subset(testinds)
        val_data = self.subset(valinds)
        return train_data, test_data, val_data

    # this adds in local density estimates from ixnos to our ribotransformer
    # object
    def add_ixdata(self, ixmodel):
        # get relevant parameters from the rdata object
        n_genes = self.data['codons'].shape[0]
        n_pos = self.data['codons'].shape[2]
        self.data['ixnos'] = torch.zeros(int(n_genes), int(n_pos))
        self.data['ixnos'] += -1
        alltrs = list(self.gene2num.index)
        chunk_size = 1000
        trchunks = [alltrs[i:i + chunk_size]
                    for i in range(0, len(alltrs), chunk_size)]

        #read input files again for the ixnos object
        psitecovtbl = pd.read_csv(
            self.inputfile
        )
        assert psitecovtbl.shape[0] > 1000
        cdsdims = pd.read_csv(self.inputcdsfile)
        cdsdims = cdsdims.set_index('tr_id')
        psitecovtbl = trim_middle(psitecovtbl, cdsdims, RPAD, NTOKS)
        print('calculating ribosome density locally with ixnos')
        for trchunk in trchunks:
            print('.')
            assert pd.Series(trchunk).isin(psitecovtbl.tr_id).all()
            # create an ixnos object, including the start/end proximal sites
            cpsitecovtbltrim = psitecovtbl[psitecovtbl.tr_id.isin(trchunk)]
            n_trs = len(cpsitecovtbltrim.tr_id.unique())
            ch_cdsdims = cdsdims[cdsdims.index.isin(trchunk)]
            ch_cdsdims = cdsdims.reset_index()
            allposixdata = Ixnosdata(cpsitecovtbltrim, ch_cdsdims,
                                     fptrim=0, tptrim=0, tr_frac=1,
                                     top_n_thresh=n_trs)
            with torch.no_grad():
                ixpreds = ixmodel(allposixdata.X_tr.float()).detach()
            # get the cumulative bounds of these from the sizes
            cbnds = np.cumsum([0] + allposixdata.bounds_tr)
            bndpairs = zip(cbnds, cbnds[1:])
            # using the bounds, chunk our predictions
            cpredlist = [
                ixpreds[lft:r] for lft, r in bndpairs]
            assert pd.Series(allposixdata.traintrs).isin(
                self.gene2num.index).all()
            # for each transcript, add in our local density estimate
            for i in range(len(allposixdata.traintrs)):
                itr = allposixdata.traintrs[i]
                trind = self.gene2num[itr]
                chk_len = 5 + len(cpredlist[i])
                cpreds = cpredlist[i].reshape([-1])
                self.data['ixnos'][trind, 5:chk_len] = cpreds
        # shape into correct dims for ribotransformer
        self.data['ixnos'] = self.data['ixnos'].detach()
        self.data['ixnos'] = self.data['ixnos'].reshape([-1, 1, 512])

    def datadim(self):
        return next(iter(self))[0][0].shape[1]

    def __iter__(self):
        if(self.batchshuffle):
            indices = torch.randperm(len(self))
        else:
            indices = list(range(len(self)))

        for batchind in range(0, len(self), self.batch_size):
            seq_len = min(self.batch_size, len(self) - batchind)
            batchinds = indices[batchind:batchind + seq_len]
            batch = self.get_batch(batchinds)
            yield batch

    def __len__(self):
        return len(self.gene2num)


###############################################################################
if __name__ == '__main__':
    sys.argv = ['']
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input_psites_df",
                        default='../eif4f_pipeline/pipeline/ribotransdata/'
                        'negIAA/negIAA.all.psites.csv')
    parser.add_argument("-c", help="input cds dims df",
                        default='../eif4f_pipeline/pipeline/ribotransdata/'
                        'negIAA/negIAA.cdsdims.csv')
    parser.add_argument(
        "-m",
        help="IxnosTorch model object for elongation calculations",
        default='ribotranstest.ixmodel_best.pt'
    )
    parser.add_argument("-o", help="ribotrans_test", default="ribotranstest")
    args = parser.parse_args()
    NTOKS = 512
    RPAD = 5
    model_file = Path(args.m)
    # make our ribotransformer data object, trim cds
    rdata = RiboTransData(args.i, args.c, NTOKS, RIBODENS_THRESH, rpad=RPAD)
    # make our ixnos model from saved parameters
    ixmodel = Net.from_state_dict(torch.load(model_file)['beststate'])
    # add the ixnos local elongation rates
    rdata.add_ixdata(ixmodel)
    # save our object
    torch.save(rdata, "data/" + args.o + "_rdata.pt")
