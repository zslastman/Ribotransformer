from pathlib import Path,PurePath 


#parse transcript names out of the esmfile names
esmweightfolder = '/fast/scratch/users/dharnet_m/tmp/yeast_tokens/'
esmfiles = glob.glob(esmweightfolder+'/*')
issplitheader = PurePath(esmfiles[0]).name.count('|')>3
esmtrs =np.array([PurePath(f).name.split('|')[0] for f in esmfiles])
esmfiles = pd.Series(esmfiles,index=esmtrs)

esmtrs2get = [g for g in ugenes if g in esmtrs]


for i,esmfile in enumerate(esmfiles[esmtrs2get[0:3]]):
    if not i % 10: print('.')
    pass

esmdat = torch.load(esmfile)['representations'][34]

    esmdat = esmdat[0:(512-10),:]
    esmtensor[i,:,10:(esmdat.shape[0]+10)] = esmdat.transpose(0,1)

cdsdim = cdsdims[cdsdims.index==esmtrs2get[i]]
start = cdsdim.aug
stop = cdsdim.stop

firsthalfend = np.ceil((trcodons/2)).clip(upper=NTOKS/2)-NFLANK#1based
# firsthalfend[tr]
sechalfstart = (trcodons-NFLANK) - np.floor((trcodons/2)).clip(upper=NTOKS/2)



################################################################################
########ESM data
################################################################################






tdata[0][1]

idx = gene2num.index[1:4]

srdata = rdata.subset(idx)
srdata.ribosignal.shape

#now let's add the tensor data
tokentensorfile = 'yeast_test_esmtensor.pt'
esmgenes,esmtensor = torch.load(open(tokentensorfile,'rb'))

tensname='esmtokens'
srdata.add_tensor(esmgenes,esmtensor,tensname)
    # self=

esmrdata.get_batch()



# def add_tokens(rdata,ugenes,esmtensor):


add_tokens(rdata,ugenes,esmtensor)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bptt=100



print('try new tpm method')
ribodens = ribodens[ugenes].values
ribodens = ribodens**-1
ribodens = torch.FloatTensor(ribodens).log()



if GET_ESM_TOKENS:
    assert len(esmfiles)>0
    if not 'esmtensor' in globals().keys():
        try:
            ugenes,esmtensor = torch.load(open(ribodatafile.replace('.csv','_')+'esmtensor.pt','rb'))
            n_genes = ugenes.shape[0]

        except FileNotFoundError:

            ugenes = psitecovtbl.gene.unique()
            ugenes.sort()

            if 'yeast' in ribodatafile:
                ugenetrs=pd.Series(ugenes)
            else:
                ugenetrs = pd.Series(ugenes).str.extract('(ENSTR?\\d+)')
                foo #fix this so it's a simple series
      
            hasesm = ugenetrs.isin(esmfiles.index)
            # ugenes = ugenes[hasesm.values[:,0]]
            ugenes = ugenes[hasesm.values]
            ugenetrs = ugenetrs[hasesm.values]
            ugeneesmfiles = esmfiles[ugenetrs.values]

            n_genes = ugenes.shape[0]

            print('parsing ESM data')
            esmtensor = torch.zeros([n_genes,n_toks,n_len])

            for i,esmfile in enumerate(ugeneesmfiles):
                if not i % 10: print('.')
                esmdat = torch.load(esmfile)['representations'][34]
                esmdat = esmdat[0:(512-10),:]
                esmtensor[i,:,10:(esmdat.shape[0]+10)] = esmdat.transpose(0,1)
        
            torch.save([ugenes,esmtensor],  ribodatafile.replace('.csv','_')+'esmtensor.pt')

    ugenesinreads = pd.Series(ugenes).isin(psitecovtbl.gene)
    ugenes = ugenes[ugenesinreads]
    esmtensor = esmtensor[ugenesinreads]
    psitecovtbl = psitecovtbl[psitecovtbl.gene.isin(ugenes)]
    n_genes = ugenes.shape[0]   
    assert esmtensor.shape == torch.Size([n_genes,n_toks,n_len])

else:    
    ugenes = psitecovtbl.gene.unique()
    ugenes.sort()









def cget_batch(source, targetsource, offsetsource, i, bptt=100,device=device):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # target = targetsource[i:i+seq_len]
    target = targetsource[i:i+seq_len].unsqueeze(1)
    # offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
    offset = offsetsource[i:i+seq_len].unsqueeze(1).unsqueeze(1)
    return data.to(device), target.to(device), offset.to(device)

assert [x.shape for x in cget_batch(codons, ribosignal, ribodens, 1)] ==  [
     torch.Size([bptt, n_cods, n_len ]),
     torch.Size([bptt, 1, n_len]),
     torch.Size([bptt, 1, 1])]


print('splitting...')
allinds = np.array(random.sample(range(0,n_genes),k=n_genes))
traininds = allinds[0:int(n_genes*.6)]
testinds = allinds[(int(n_genes*.6)):int(n_genes*.8)]
valinds = allinds[int(n_genes*.8):]

esmmax = esmtensor.max(0).values.max(1).values
esmmin = esmtensor.min(0).values.min(1).values

esmmax = esmmax.reshape([1,n_toks,1])
esmmin = esmmin.reshape([1,n_toks,1])

assert not  (esmmax == esmmin).any()

esmtensornorm = (esmtensor - esmmin) / (esmmax - esmmin )




################################################################################
########H5 encoding of data
################################################################################
import h5py
import os
esmtensororig = esmtensor

h5file = 'testfile.h5'
testfile = h5py.File(h5file, 'w')
os.remove(h5file)


testfile['data'] = torch.eye(1000)

testfile['data'] = ribosignal


testfile['data'][0:100,0:100]
testfile['data'][0:100,0:100]


testfile['ribosignal'] = ribosignal

testfile['ribosignal'].resize(testfile['ribosignal'].shape[0]+10**4, axis=0)

h5ob = h5py.File(h5file, 'r')
torch.from_numpy(np.array(h5ob['data'][0:3,0:3]))


with h5py.File(path, "a") as f:
    dset = f.create_dataset('ribosignal', (10**5,), maxshape=(None,),
                            dtype='i8', chunks=(10**4,))
    dset[:] = np.random.random(dset.shape)        
    print(dset.shape)
    # (100000,)

    for i in range(3):
        dset.resize(dset.shape[0]+10**4, axis=0)   
        dset[-10**4:] = np.random.random(10**4)
        print(dset.shape)
        # (110000,)
        # (120000,)
        # (130000,)




class HDF5DatasetSingle(data.Dataset):
    """Represents an abstract HDF5 dataset that's in a single file.
    
    Input params:
        file_path: Path to the hdf5 file.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))
    



