
if True:
    rdata.usedata = ['codons', 'seqfeats']
    tdata = next(iter(rdata))
    assert rdata.datadim() == 80

    assert type(rdata.offset) is torch.Tensor
    assert list(tdata[0][0].shape) == [
        rdata.batch_size, rdata.datadim(), rdata.n_toks]
    assert list(tdata[0][1].shape) == [rdata.batch_size, 1, 1]

    assert not (next(iter(rdata))[1][0] == next(iter(rdata))[1][0]).all()
    rdata.usedata = ['codons']

    assert tdata[1].shape == rdata.orfclip(tdata[0][0]).shape

if True:
    idx = rdata.gene2num.index[1:4]
    srdata = rdata.subset(idx)

    srdata.ribosignal.shape
    srdata.batch_size = 3
    tdata = next(iter(srdata))
    assert list(tdata[0][0].shape) == [3, rdata.datadim(), rdata.n_toks]
    assert list(tdata[0][1].shape) == [3, 1, 1]
    rdata.batchshuffle = False
    assert (next(iter(rdata))[0][0] == next(iter(rdata))[0][0]).all()
    rdata.batchshuffle = True
    assert not (next(iter(rdata))[0][0] == next(iter(rdata))[0][0]).all()


def split_data_vonly(rdata, valsize=1000):
    n_genes = len(rdata)

    allinds = np.array(random.sample(range(0, n_genes), k=n_genes))
    tvsize = valsize + tsize
    traininds = rdata.gene2num.index[allinds[0:(len(allinds) - tvsize)]]
    testinds = rdata.gene2num.index[allinds[(
        len(allinds) - tvsize):(len(allinds) - valsize)]]
    valinds = rdata.gene2num.index[allinds[(len(allinds) - valsize):]]

    train_data = rdata.subset(traininds)
    test_data = rdata.subset(testinds)
    val_data = rdata.subset(valinds)
    return train_data, test_data, val_data


train_data, test_data, val_data = split_data(rdata)

riboob = train_data


def get_codstrengths(riboob):
    ribosignal = riboob.ribosignal
    codons = riboob.data['codons']
    n_cods = codons.shape[1]
    ribodens = riboob.offset.squeeze(2)
    codstrengths = [
        (ribosignal / ribodens)[riboob.data['codons']
                                [:, i, :] == 1].mean().item()
        for i in range(0, n_cods)]
    codstrengths = torch.FloatTensor(codstrengths[1:])
    return codstrengths


codstrengths = get_codstrengths(rdata)


if False:

    # consistent codon strengths from train to val data
    txtplot(
        np.array(get_codstrengths(train_data)),
        np.array(get_codstrengths(val_data))
    )

    # consistent ribosome densities with pandas method
    psitecovtbltrimused = psitecovtbltrim[psitecovtbltrim.tr_id.isin(
        rdata.ugenes)]

    # gene offsets look like they should
    plx.clear_plot()
    plx.scatter(
        np.array(psitecovtbltrimused.groupby('tr_id')[
                 'ribosome_count'].mean()[rdata.ugenes].values).squeeze(),
        np.array(rdata.offset).squeeze(),
        np.array(rdata.offset).squeeze(),
        rows=17, cols=70)
    plx.show()

    # codon level stats are same as using pandas

    traintrsums = psitecovtbltrimused.groupby(
        'tr_id')['ribosome_count'].sum().reset_index()

    traintrsums = traintrsums.rename({'ribosome_count': 'tr_count'}, axis=1)
    psitecovtbltrimused = psitecovtbltrimused.merge(trsums)
    psitecovtbltrimused['dens'] = psitecovtbltrimused['ribosome_count'] / \
        psitecovtbltrimused['tr_count']
    #
    rblcstrengths = psitecovtbltrimused.groupby('codon')['dens'].mean()
    rblcstrengths = rblcstrengths.reset_index()
    rblcstrengths['num'] = rdata.codonnum[rblcstrengths['codon']].values
    rblcstrengths = rblcstrengths.sort_values('num')
    rblcstrengths = rblcstrengths.dens.values
    #
    plx.clear_plot()
    plx.scatter(
        np.array(get_codstrengths(train_data)),
        rblcstrengths,
        rows=17, cols=70)
    plx.show()


if True:
    # we can also just fake rdata
    # rdata = oldrdata
    rdata_orig = copy.deepcopy(rdata)
    fakerdata = copy.deepcopy(rdata_orig)
    fcodstrengths = torch.cat([ten([0]), codstrengths])
    fsignal = (fcodstrengths.reshape([1, 65, 1])
               * fakerdata.data['codons']).sum(axis=1)
    fsignal = fsignal * fakerdata.offset.reshape([-1, 1])

    # poseffects = fakerdata.ribosignal.mean(axis=0).reshape([1,-1])
    # fsignal = fsignal*poseffects

    fakerdata.ribosignal = fsignal

    cdslens = (fakerdata.data['codons'][:, 0, :] != 1).sum(1)
    frac_filled = (cdslens / 512.0)

    fakerdata.n_toks = NTOKS
    fakerdata.offset = (fakerdata.ribosignal.mean(
        axis=1) / frac_filled).reshape([len(fakerdata.ugenes), 1])
    fakerdata.offset = fakerdata.offset.reshape([-1, 1, 1])

    fakecodstrengths = get_codstrengths(fakerdata)
    # indeed this works.
    txtplot(fakecodstrengths, codstrengths)
    assert fakerdata.ribosignal.shape == torch.Size(
        [rdata.ribosignal.shape[0], NTOKS])
    # yup, this also works
    txtplot(fakerdata.ribosignal.sum(axis=0))

    fakerdata.offset == rdata.offset


if False:
    s_codstrengths = pd.Series(codstrengths, index=pd.Series(
        rdata.codonnum.index, name='codon'), name='mydens')
    s_codstrengths = pd.DataFrame(
        {'codon': rdata.codonnum.index, 'mydens': codstrengths})
    wbergsupp = pd.read_csv(
        '../cortexomics/ext_data/weinberg_etal_2016_S2.tsv', sep='\t')
    wbergsupp = wbergsupp[['Codon', 'RiboDensity at A-site']]
    wbergsupp.columns = ['codon', 'wberg_dens']
    dtcompdf = (rdata.codonnum.
                reset_index().
                rename(columns={'index': 'codon', 0: 'num'}).
                merge(wbergsupp).merge(s_codstrengths)
                )
    txtplot(np.log(dtcompdf.wberg_dens), np.log(dtcompdf.mydens))

    rdata.batch_size = 20
    rdata.usedata = ['codons']
    train_data, test_data, val_data = split_data(rdata, 500, 1)
    # train_data,test_data,val_data = split_data(fakerdata,500,1)

    assert train_data.batchshuffle

    def grouper(n, iterable, fillvalue=None):
        "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return it.zip_longest(fillvalue=fillvalue, *args)

    # I want to confirm that when I clip my ribodata object and when I look at ixnos data I have
    # the exact same data
    # To do this I"ll get a single gene, of length less than 512
    # Then I can figure out the total counts for that gene in the ixnos object
    # then I can figure out the total counts when I clip the ribodata object
    testtrs = (psitecovtbl.groupby('tr_id').size().index)[0:2]
    testtr = 'ENST00000000233.10'
    trdata = rdata.subset(testtrs)

    testpsitecovtbl = psitecovtbl.query('tr_id.isin(@testtrs)')
    psitecovtbl.query('tr_id.isin(@testtrs)').ribosome_count.sum()
    trdata.ribosignal[0, :].sum()
    trdata.ribosignal[1, :].sum()
    psitecovtbl.query('tr_id==@testtrs[0]').ribosome_count.sum()
    psitecovtbl.query('tr_id==@testtrs[1]').ribosome_count.sum()
    psitecovtbltrim.query('tr_id==@testtrs[0]').ribosome_count.sum()
    psitecovtbltrim.query('tr_id==@testtrs[1]').ribosome_count.sum()

    psitecovtbl.query('tr_id==@testtrs[1]')
    psitecovtbltrim.query('tr_id==@testtrs[1]')

    ch_cdsdims = cdsdims[cdsdims.tr_id.isin(testtrs)]
    exec(
        open('src/IxnosTorch/train_ixmodel.py').read(),
        train_ixmodel)

    allposixdata = Ixnosdata(testpsitecovtbl, ch_cdsdims,
                             fptrim=0, tptrim=0, tr_frac=1, top_n_thresh=2)

    allposixdata.y_tr
    trdata.ribosignal[0]
    allposixdata.y_tr[0:100]
    trdata.ribosignal[1]
    psitecovtbltrim.query('tr_id==@testtrs[1]').tail(10)
    allposixdata.bounds_tr
    cbounds = np.cumsum([0] + allposixdata.bounds_tr)
    allposixdata.y_tr[cbounds[0]:cbounds[0 + 1]]
    allposixdata.y_tr[cbounds[1]:cbounds[1 + 1]]
    allposixdata.y_tr[cbounds[0 + 1] - 1:cbounds[0 + 1] + 3]
    # yup these bounds work okay so we have lal the cds in the ixxons data here

    # okay so it looks like that ixnos data contains up to the stop codon
    # And from the start codon
    # so does clipping our object get us the same sites?
    cliptensor = trdata.orfclip(trdata.data['codons'], 0, 0)
    # okay so this the full CDS
    trdata.ribosignal[1][cliptensor[1, :]]
    # this doesn't even look like the right length
    len(trdata.ribosignal[1][cliptensor[1, :]])
    trdata.ribosignal[1][cliptensor[1, :]]
    # can't make sense of it at all
    psitecovtbltrim.query('tr_id==@testtrs[0]').head(20)
    psitecovtbltrim.query('tr_id==@testtrs[0]').tail(20)
    allposixdata.y_tr[cbounds[0]:cbounds[0 + 1]]

    psitecovtbltrim.query('tr_id==@testtrs[1]').tail(20)
    # uuuuuh not sure if the clip tensor is right at all
    # okay so this uniquely identifieds our second test tr
    ((rdata.ribosignal[:, 0] == 15) & (rdata.ribosignal[:, 1] == 1) & (
        rdata.ribosignal[:, 2] == 3) & (rdata.ribosignal[:, 3] == 0)).sum()
    # looks like it's 1073 in th original object
    np.where(((rdata.ribosignal[:, 0] == 15) & (rdata.ribosignal[:, 1] == 1) & (
        rdata.ribosignal[:, 2] == 3) & (rdata.ribosignal[:, 3] == 0)))
    # this has 287 entries if we expand 5nt on either side or 277
    psitecovtbltrim.query('tr_id==@testtrs[1]')
    # and 190 or 180
    psitecovtbltrim.query('tr_id==@testtrs[0]')

    # this is the correct data for it in the rdata object
    NTOKS - rdata.data['codons'][1073, 0, :].sum()
    # so what about our trdata object - it's 1 of course
    np.where(((trdata.ribosignal[:, 0] == 15) & (trdata.ribosignal[:, 1] == 1) & (
        trdata.ribosignal[:, 2] == 3) & (trdata.ribosignal[:, 3] == 0)))
    NTOKS - trdata.data['codons'][1, 0, :].sum()
    rdata.orfclip(trdata.data['codons'], 0, 0).sum(axis=1)
    # orf clipping gives us the right amount for our two tests

    # it's the chunkpredlist thing that's wrong

    # actually this cilpping is working. 0 gives us all coding bases, no stop, 1 and -1 clips one off either end
    trdata.ribosignal[1][cliptensor[1, :]].sum()

    rdata.data['ixnos'][1073, :, 5:(5 + 287)]
    # preeictions below are too long...
