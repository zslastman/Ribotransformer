from multiprocessing import Pool
import torch
import argparse
import sys
import numpy as np
import pandas as pd
from seqfold import dg
from pathlib import Path
sparse = torch.sparse_coo_tensor
fl = torch.sparse.FloatTensor  # type: ignore[attr-defined]
ten = torch.tensor
nts = 'ACGT'
cods = [a + b + c for a in nts for b in nts for c in nts]

foo = torch.cat


class Ixnosdata:
    nts = 'ACGT'
    codinds = list(range(len(cods)))
    cod2id = pd.Series(dict(zip(cods, codinds)))

    def get_seq_X(self, tr: str):
        tr_coddf = (
            self.df.
            query('tr_id==@tr')
        )
        nvals = tr_coddf.shape[0]
        allcodvals = torch.sparse_coo_tensor(
            torch.LongTensor([  # type: ignore[call-arg]
                np.array(range(nvals)),
                self.cod2id[tr_coddf.codon].values
            ]),
            torch.LongTensor([1] * nvals),  # type: ignore[call-arg]
            torch.Size([nvals, 64]),
            dtype=torch.float64
        ).to_dense()
        n_inner = nvals - 9
        # mydatseq = X_to_seq(allcodvals)
        # xtrseq = X_to_seq(np.concatenate([X_tr[0:n_inner,0:64]]))
        # extraseq = X_to_seq(X_tr[n_inner-9:n_inner,0+(64*9):64+(64*9)])
        # assert mydatseq == xtrseq+extraseq
        # okay great so I have all the sequence there in their X_tr
        codmat = torch.cat([allcodvals[0 + i:i + n_inner, :]
                            for i in range(0, 10)], dim=int(1))
        # assert (np.array(1.0*codmat) == X_tr[0:167,0:640]).all()
        # YES okay so I can recreate their data!
        nuc2id = pd.Series(dict(zip(list(Ixnosdata.nts), list(range(4)))))
        nvals = tr_coddf.shape[0]
        i = 0
        nucvals = []
        for i in range(3):
            nucvals.append(
                torch.sparse_coo_tensor(
                    ten([
                        np.array(range(nvals)),
                        nuc2id[tr_coddf.codon.str[i]].values
                    ]),
                    ten([(1)] * nvals, dtype=torch.float64),
                    torch.Size([nvals, 4])
                ).to_dense()
            )
        nuctensor = torch.stack(nucvals)
        # assert nucvals.shape==(3,176,4)
        nuctensor = nuctensor.permute([2, 0, 1])
        # assert nuctensor.shape==(4,3,176)
        nuctensor = nuctensor.permute([1, 0, 2]).reshape([12, nvals])
        rn = range(0, 10)
        nuctensor = torch.cat([nuctensor[:, 0 + i:i + n_inner] for i in rn])
        # assert nuctensor.shape==(120,167)
        seqvals = torch.cat([codmat.T, nuctensor])
        seqvals = seqvals.T
        #
        yvals = tr_coddf[5:-4].ribosome_count.values
        yvals = ten(yvals, dtype=torch.float64)
        yvals = yvals / yvals.mean()
        sys.stdout.write('.')
        return(seqvals, yvals)

    def __init__(self,
                 df: pd.DataFrame,
                 cdsdims: pd.DataFrame,
                 fptrim: int = 20, tptrim: int = 20, fwidth: int = 5,
                 countmeanthresh: int = None,
                 top_n_thresh: int = 500,
                 get_energy: bool = False,
                 countfilt: bool = True):
        cdsdims['n_cod'] = (cdsdims['stop'] - cdsdims['aug']) / 3
        cdsdims = cdsdims.loc[(cdsdims['n_cod'] % 1 == 0)]
        cdsdims = cdsdims.loc[cdsdims['n_cod'] > (fptrim + tptrim)]
        self.df: pd.DataFrame = (
            df.merge(cdsdims).
            query('codon_idx>=(@fptrim-@fwidth)').
            query('codon_idx <= (n_cod-(@tptrim-@fwidth+1))')
        )
        trcounts = self.df.groupby('tr_id')['ribosome_count'].mean()
        if countfilt:
            if top_n_thresh is not None:
                trcounts = trcounts[trcounts != 0]
                alltrs = (trcounts.sort_values(ascending=False).
                          head(top_n_thresh).index.values)
            else:
                alltrs = trcounts[trcounts > countmeanthresh].index
        else:
            alltrs = self.df.tr_id.unique()
        traintrs = np.random.choice(alltrs, int(len(alltrs) * 0.75))
        testrs = np.array(list(set(alltrs) - set(traintrs)))
        #
        self.my_data_tr = [self.get_seq_X(tr) for tr in traintrs]
        self.my_data_te = [self.get_seq_X(tr) for tr in testrs]
        #
        self.X_tr: torch.Tensor = torch.cat([x[0] for x in self.my_data_tr])
        self.y_tr: torch.Tensor = torch.cat([x[1] for x in self.my_data_tr])
        self.X_te: torch.Tensor = torch.cat([x[0] for x in self.my_data_te])
        self.y_te: torch.Tensor = torch.cat([x[1] for x in self.my_data_te])
        assert self.X_tr.shape[0] == self.y_tr.shape[0]
        assert self.X_te.shape[0] == self.y_te.shape[0]
        self.bounds_tr = [x[0].shape[0] for x in self.my_data_tr]
        self.bounds_te = [x[0].shape[0] for x in self.my_data_te]
        self.fwidth: int = fwidth
        del self.df

        if get_energy:
            ENERGYCILP = 20
            #
            with Pool(4) as p:
                struct_tr = p.map(self.get_energyvals, traintrs)
                # pickle.dump(struct_t. r, open(f'struct_tr.pkl', 'wb'))
                structmat_tr = torch.cat([ten(x[2][:-3]) for x in struct_tr])
                structmat_tr = structmat_tr.reshape([3, -1])
                structmat_tr = structmat_tr.T
                structmat_tr = torch.clamp(
                    structmat_tr, -ENERGYCILP, ENERGYCILP)
                assert structmat_tr.shape[0] == self.X_tr.shape[0]

            #
            with Pool(4) as p:
                struct_te = p.map(self.get_energyvals, testrs)
                # pickle.dump(struct_te, open(f'struct_te.pkl', 'wb'))
                # type: ignore[call-arg]
                structmat_te = torch.cat([ten(x[2][:-3]) for x in struct_te])
                structmat_te = structmat_te.reshape([3, -1])
                structmat_te = structmat_te.T
                structmat_te = torch.clamp(
                    structmat_te, -ENERGYCILP, ENERGYCILP)
                assert structmat_te.shape[0] == self.X_te.shape[0]

                self.X_tr = torch.cat([1.0 * self.X_tr, structmat_tr], dim=1)
                self.X_te = torch.cat([1.0 * self.X_te, structmat_te], dim=1)
        self.df = df

    def X_to_seq(self, X):
        assert X.shape[1] == 4
        seqfromdat = ''.join([self.id2cod[cod] for cod in np.where(X)[1]])
        return(seqfromdat)

    def get_energyvals(self, tr, n_bps=30):
        tr_coddf = (
            self.df.
            query('tr_id==@tr')
        )
        nvals = tr_coddf.shape[0]
        n_inner = nvals - self.fwidth + 1
        all_e_seq = ''.join(tr_coddf.codon).replace('T', 'U')[1:]
        e_vals = [dg(all_e_seq[i:][:n_bps]) for i in range(0, n_inner * 3)]
        # flat_Xevails = X_tr[:,-3:].reshape([-1])
        # txtplot(np.array(e_vals),flat_Xevails[:167*3])
        # stats.pearsonr(np.array(e_vals),flat_Xevails[:167*3])
        sys.stdout.write('.')
        return([all_e_seq, n_bps, e_vals])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="Input dataframe with read counts per codon",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default=f'cortexomicstest.all.psites.csv'
    )
    parser.add_argument(
        "-c",
        help="Input dataframe with dimensions of csv",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default=f'cortexomicstest.cdsdims.csv'
    )
    args = parser.parse_args()

    df_file: Path = Path(args.i)
    cdsdims_file: Path = Path(args.c)
    df: pd.DataFrame = pd.read_csv(df_file)
    cdsdims: pd.DataFrame = pd.read_csv(cdsdims_file)
    # df = df.loc[range(0, 10000)]

    # create dataset
    ixdataset = Ixnosdata(df, cdsdims)
