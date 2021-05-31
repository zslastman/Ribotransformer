import argparse
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="Input dataframe with read counts per codon",
        # default="pipeline/weinberg_yeast_riboAligned.out.sort.bam"
        default=f'cortexomicstest.all.psites.csv'
    )
  
    args = parser.parse_args()

    df_file: Path = Path(args.i)
    df: pd.DataFrame = pd.read_csv(df_file, sep='\t')

###############################################################################
########
###############################################################################

allpsites = pd.read_csv(f'ribotranstest.all.psites.csv')
ixnostr_trs = pd.read_csv('/fast/AG_Ohler/dharnet/Ribotransformer/iXnos/expts/weinberg/process/tr_set_bounds.size.27.31.trunc.20.20.min_cts.200.min_cod.100.top.500.txt',
    sep='\t',
    header=None
  )[0]
ixnostr_trs=sorted(ixnostr_trs)
ixnoste_trs = pd.read_csv('/fast/AG_Ohler/dharnet/Ribotransformer/iXnos/expts/weinberg/process/te_set_bounds.size.27.31.trunc.20.20.min_cts.200.min_cod.100.top.500.txt',
    sep='\t',
    header=None
  )[0]
ixnoste_trs=sorted(ixnoste_trs)
assert 'tr_id' in psitestest.columns

import pandas as pd
import torch
ten = torch.Tensor

alpha = 'ACGT'
cods = [a+b+c for a in alpha for b in alpha for c in alpha]
codinds = list(range(len(cods)))
cod2id = pd.Series(dict(zip(cods,codinds)))


psitestest = sumpsites.merge(cdsdims).query('tr_id=="YAL003W"').query('codon_idx>=(20-5)').query('codon_idx <= (n_cod-(20-4))')
sumpsites

def X_to_seq(X):
    alpha="ACGT"
    codons = [x+y+z for x in alpha for y in alpha for z in alpha]
    cod2id = {codon:idx for idx, codon in enumerate(codons)}
    id2cod = dict(zip(cod2id.values(),cod2id.keys()))
    assert X.shape[1]==4
    seqfromdat = ''.join([id2cod[cod] for cod in np.where(X)[1]])
    return(seqfromdat)

tr = ixnostr_trs[1]
tr = 'YAL003W'

def get_seq_X(tr):
  tr_coddf = sumpsites.merge(cdsdims).query('tr_id==@tr').query('codon_idx>=(20-5)').query('codon_idx <= (n_cod-(20-4))')
  nvals = tr_coddf.shape[0]
  allcodvals = torch.sparse.FloatTensor(
      torch.LongTensor([
             np.array(range(nvals)),
             cod2id[tr_coddf.codon].values
      ]),
      torch.LongTensor([1]*nvals),
      torch.Size([nvals, 64])
  ).to_dense()
  n_inner = nvals - 9
  # mydatseq = X_to_seq(allcodvals)
  # xtrseq = X_to_seq(np.concatenate([X_tr[0:n_inner,0:64]]))
  # extraseq = X_to_seq(X_tr[n_inner-9:n_inner,0+(64*9):64+(64*9)])
  # assert mydatseq == xtrseq+extraseq
  #okay great so I have all the sequence there in their X_tr
  codmat = torch.cat([allcodvals[0+i:i+n_inner,:] for i in range(0,10)],axis=1)
  # assert (np.array(1.0*codmat) == X_tr[0:167,0:640]).all()
  #YES okay so I can recreate their data!
  nuc2id = pd.Series(dict(zip(list(alpha),list(range(4)))))
  nvals = tr_coddf.shape[0]
  i=0
  nucvals = []
  for i in range(3):
    nucvals.append(
      torch.sparse.FloatTensor(
          torch.LongTensor([
                 np.array(range(nvals)),
                 nuc2id[tr_coddf.codon.str[i]].values
          ]),
          torch.LongTensor([1]*nvals),
          torch.Size([nvals, 4])
      ).to_dense()
    )
  nucvals = torch.stack(nucvals)
  # assert nucvals.shape==(3,176,4)
  nucvals= nucvals.permute([2,0,1])
  # nucvals = torch.permute(nucvals,[0,1,2],[1,2,0])
  # assert nucvals.shape==(4,3,176)
  nucvals = nucvals.permute([1,0,2]).reshape([12,nvals])
  nucvals = torch.cat([nucvals[:,0+i:i+n_inner] for i in range(0,10)])
  # assert nucvals.shape==(120,167)
  seqvals = torch.cat([codmat.T,nucvals])
  seqvals = seqvals.T
  # assert np.array(nucvals.T).shape==X_tr[0:167,640:-3].shape
  # assert (np.array(nucvals.T)==X_tr[0:167,640:-3]).all()
  #Feature matrix recreated!
  # assert ((1.0*np.array(seqvals))==X_tr[0:167,0:760]).all()
  #
  yvals = ten(tr_coddf[5:-4].ribosome_count.values)
  yvals = yvals / yvals.mean()
  print('.')
  return(seqvals,yvals)

my_data_tr = [get_seq_X(tr) for tr in ixnostr_trs]
my_X_tr = torch.cat([x[0] for x in my_data_tr])
my_y_tr = torch.cat([x[1] for x in my_data_tr])
my_data_te = [get_seq_X(tr) for tr in ixnoste_trs]
my_X_te = torch.cat([x[0] for x in my_data_te])
my_y_te = torch.cat([x[1] for x in my_data_te])

assert mlp_X.shape[0]==mlp_y.shape[0]



# (my_X_tr[0:167]==ten(X_tr[0:167,:-3])).all()
# (my_X_tr[167:167+100]==ten(X_tr[167:167+100,:-3])).all()

# txtplot(my_y_tr[0:167],ten(y_tr[0:167]))
# (my_y_tr[0:167]==ten(y_tr[0:167])).all()
# (my_y_tr[167:167+100]==ten(y_tr[167:167+100,:-3])).all()


#get relevant data minus a codon
#'UAAACGCUUCUUUGGCUGACAAGUCAUACA'


#so this takes a long fucking time.
#I should write a seperate script that produces the
#set of energy values for a given sequence, footprints size

def get_energyvals(tr,n_bps=30):
  tr_coddf = (sumpsites.merge(cdsdims).
      query('tr_id==@tr').
      query('codon_idx>=(20-5-1)').
      query('codon_idx <= (n_cod-(20-4))'))
  nvals = tr_coddf.shape[0]
  n_inner = nvals - 9 
  all_e_seq = ''.join(tr_coddf.codon).replace('T','U')[1:]
  e_vals = [dg(all_e_seq[i:][:n_bps]) for i in range(0,n_inner*3)]
  # flat_Xevails = X_tr[:,-3:].reshape([-1])
  # txtplot(np.array(e_vals),flat_Xevails[:167*3])
  # stats.pearsonr(np.array(e_vals),flat_Xevails[:167*3])
  sys.stdout.write('.')
  return([all_e_seq,n_bps,e_vals])
  # return ten(evals).reshape([3,-1]).T

#
import pickle
e_vals_tr = [get_energyvals(tr) for tr in ixnostr_trs]
pickle.dump(e_vals_tr,open(f'ixnos_tr_evals.pkl','wb'))
e_mat_tr = torch.cat([x for x in e_vals_tr])

e_vals_te = [get_energyvals(tr) for tr in ixnoste_trs]
e_mat_te = torch.cat([x for x in e_vals_te])
pickle.dump(e_vals_te,open(f'ixnos_te_evals.pkl','wb'))

from multiprocessing import Pool 
#
with Pool(4) as p:
  struct_tr = p.map(get_energyvals,ixnostr_trs)
pickle.dump(struct_tr,open(f'struct_tr.pkl','wb'))
structmat_tr = torch.cat([ten(x[2][:-3]) for x in struct_tr])
structmat_tr = structmat_tr.reshape([3,-1])
structmat_tr = structmat_tr.T
structmat_tr = torch.clamp(structmat_tr,-20,20)
assert structmat_tr.shape[0] == my_X_tr.shape[0]

#
with Pool(4) as p:
  struct_te = p.map(get_energyvals,ixnoste_trs)
pickle.dump(struct_te,open(f'struct_te.pkl','wb'))
structmat_te = torch.cat([ten(x[2][:-3]) for x in struct_te])
structmat_te = structmat_te.reshape([3,-1])
structmat_te = structmat_te.T
structmat_te = torch.clamp(structmat_te,-20,20)
assert structmat_te.shape[0] == my_X_te.shape[0]

my_X_tr_s = torch.cat([1.0*my_X_tr,structmat_tr],axis=1)
my_X_te_s = torch.cat([1.0*my_X_te,structmat_te],axis=1)



