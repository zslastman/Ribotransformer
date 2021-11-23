
if use_ixnos:
    from src.IxnosTorch.ixnosdata import Ixnosdata
    # import src.IxnosTorch.train_ixmodel
    train_ixmodel = {}
    exec(
        open('src/IxnosTorch/train_ixmodel.py').read(),
        train_ixmodel)
    train_on_ixdata = train_ixmodel['train_on_ixdata']
    if 'level_0' not in cdsdims.columns: cdsdims=cdsdims.reset_index()
    # dataset for training
    ixdataset = Ixnosdata(psitecovtbltrim, cdsdims)
    ixdataset.y_tr
    trainres = train_on_ixdata(ixdataset, epochs=10)
    ixmodel = train_ixmodel['Net'].from_state_dict(vars(trainres)['beststate'])
    # Now - add ixnos as a feature in the rdata object
    # data set for
if True:


if False:
    cliptensor = rdata.orfclip(rdata.data['codons'])
    rdata.nribosignal = copy.deepcopy(rdata.ribosignal)
    for itr in range(rdata.ribosignal.shape[0]):
        mc = rdata.ribosignal[itr][cliptensor[itr]].mean()
        rdata.nribosignal[itr] = rdata.ribosignal[itr] / mc

    i=0
    itr = allposixdata.traintrs[i]
    itr = 'ENST00000000412.8'
    trind = rdata.gene2num[itr]
    # psitecovtbltrim.query('tr_id==@itr').head(10)
    # psitecovtbltrim.query('tr_id==@itr').tail(10)
    #okay now given we've just put the signal back in using ixnos, confirm that worked.
    icdslen = psitecovtbltrim.query('tr_id==@itr').shape[0] - 10

    rdata.ribosignal[trind,:]
    rdata.data['ixnos'][trind,:]


    clippedsig = rdata.ribosignal[trind,rdata.orfclip(rdata.data['codons'])[trind,:]]
    clippedixnos = rdata.data['ixnos'][trind,0,rdata.orfclip(rdata.data['codons'])[trind,:]]
    assert scipy.stats.pearsonr (clippedsig,clippedixnos)[0]>.9

    #okay so seq and signal match up for ixnos
    #signal for ixnos and ribotransob signal match up
    #now do ribotransob signal and ixnossignal match up?
    pos=8
    for pos in range(6,15):
        tblcod = psitecovtbltrim.query('tr_id==@itr').codon.values[pos]
        ribotrcodwhere = np.where(rdata.data['codons'][trind,:,pos])[0]
        revcodnum = pd.Series(rdata.codonnum.index,index=rdata.codonnum.values)
        ribosigcod = revcodnum[ribotrcodwhere].values
        assert ribosigcod==tblcod

if False:  # testing..

    trind=1
    i=1
    scipy.stats.pearsonr(rdata.data['ixnos'][trind, 5:(5+len(chunkpredlist[i]))],
                         rdata.ribosignal[trind, 5:(5+len(chunkpredlist[i]))])

    tcount = psitecovtbltrim.query(
        'tr_id == @itr').query('codon_idx>=0').head(482).ribosome_count

    i = 2
    itr = allposixdata.traintrs[i]
    n = len(chunkpredlist[i])
    tcount = psitecovtbltrim.query(
        'tr_id == @itr').query('codon_idx>=0').head(n).ribosome_count
    txtplot(chunkpredlist[i].flatten(), tcount)
    tcount = tcount - tcount.mean()
    tcount = tcount / tcount.std()
    txtplot(chunkpredlist[i].flatten(), tcount)
    # [len(c) for c in chunkpredlist]
    assert rdata.data['ixnos'].sum(axis=1).min() > 0



if True:
    # Load good elongation rates, does my ixos here reflect them?
    # Can I then get those out the other end of my transformer?
    goodelong = pd.read_csv(
        '../eif4f_pipeline/pipeline/ixnos_elong/negIAA/negIAA.elong.csv')
    trchunk = ixdataset.traintrs
    cpsitecovtbltrim = psitecovtbltrim[psitecovtbltrim.tr_id.isin(trchunk)]
    n_trs = len(cpsitecovtbltrim.tr_id.unique())
    ch_cdsdims = cdsdims[cdsdims.tr_id.isin(trchunk)]
    allposixdata = Ixnosdata(cpsitecovtbltrim, ch_cdsdims,
                             fptrim=0, tptrim=0, tr_frac=1, top_n_thresh=n_trs)
    allposixdata.y_tr == ixdataset.y_tr
    with torch.no_grad():
        ixpreds = ixmodel(allposixdata.X_tr.float()).detach()
    # get the cumulative bounds of these from the sizes
    cbounds = np.cumsum([0]+allposixdata.bounds_tr)
    # using the bounds, chunk our predictions
    chunkpredlist = [ixpreds[l:r] for l, r in zip(cbounds, cbounds[1:])]
    emeans = [x.mean().item() for x in chunkpredlist]
    myelong = pd.DataFrame([trchunk, pd.Series(emeans)]).transpose()
    myelong.columns = ['tr_id', 'myelong']
    myelong.myelong = pd.Series([float(x) for x in myelong.myelong.values])
    myelong = myelong.merge(goodelong)
    scipy.stats.pearsonr(myelong.myelong.values,
                         myelong.elong.values)
    trmyelong.columns = ['tr_id', 'trmyelong']
    myelong = myelong.merge(trmyelong)
    scipy.stats.pearsonr(myelong.myelong.values,
                         myelong.trmyelong.values)

    scipy.stats.pearsonr(allposixdata.y_tr,
                         [z.item() for x in chunkpredlist for z in x])

    scipy.stats.pearsonr(allposixdata.y_tr,
                         [z.item() for x in chunkpredlist for z in x])

    with torch.no_grad():
        trixpreds = ixmodel(ixdataset.X_tr.float()).detach()

    scipy.stats.pearsonr(ixdataset.y_tr,
                         trixpreds[:, 0])

    with torch.no_grad():
        trixpreds = ixmodel(ixdataset.X_tr.float()).detach()
    # do the preds from the training one match the good ones from the pipeline
    cbounds = np.cumsum([0]+ixdataset.bounds_tr)
    # using the bounds, chunk our predictions
    chunkpredlist = [trixpreds[l:r] for l, r in zip(cbounds, cbounds[1:])]
    emeans = [x.mean().item() for x in chunkpredlist]
    trmyelong = pd.DataFrame(
        [pd.Series(ixdataset.traintrs), pd.Series(emeans)]).transpose()
    trmyelong.columns = ['tr_id', 'trmyelong']
    trmyelong.trmyelong = pd.Series([float(x)
                                     for x in trmyelong.trmyelong.values])
    trmyelong = trmyelong.merge(goodelong)
    trmyelong = trmyelong[~np.isnan(trmyelong.trmyelong)]
    scipy.stats.pearsonr(trmyelong.trmyelong,
                         trmyelong.elong)
    txtplot(trmyelong.trmyelong,
            trmyelong.elong)

    # ooookay so at least these elongs are sane and match the good ones.

    # [z.item() for x in chunkpredlist for z in x])


# the stacking doesn't work