#I think this model should be able to do better

#so index 51 is the best codstrengthof the 64, so add 1
sketchdata = copy.deepcopy(train_data)
sketchdata.batch_size=1000000
tdata,ttarget=next(iter(sketchdata))
toutput = cmodel(tdata)

#indeed...
#this isn't so high
ttarget[tdata[0][:,49,:]==1].mean()
#but this is
ttarget[tdata[0][:,52,:]==1].mean()
#this isn't
ttarget[tdata[0][:,53,:]==1].mean()
#ttarget[tdata[0][:,0,:]==1].mean()this is zero
rdata.ribosignal[rdata.data['codons'][:,51,:]==1].mean()
rdata.ribosignal[rdata.data['codons'][:,52,:]==1].mean()
rdata.ribosignal[rdata.data['codons'][:,53,:]==1].mean()

txtplot(
	np.array([rdata.ribosignal[rdata.data['codons'][:,i,:]==1].mean()for i in range(0,65)])[1:],
	codstrengths
	)

#the output reflects this... a little
#this isn't so high
toutput[tdata[0][:,51,:]==1].exp().mean()
#but this is
toutput[tdata[0][:,52,:]==1].exp().mean()
#this isn't
toutput[tdata[0][:,53,:]==1].exp().mean()

toutput[tdata[0][:,51,:]==1].mean()
#but this is
toutput[tdata[0][:,52,:]==1].mean()
#this isn't
toutput[tdata[0][:,53,:]==1].mean()


#so I think I should be able to do better by increasing the corresponding
#weight.

#make the old output
toutput = cmodel(tdata)

#modify our model
newmodel = copy.deepcopy(cmodel)
newmodel.conv.weight[:,52,:] = .2
#make the new output
ntoutput = newmodel(tdata)
toutput == ntoutput
#verify they differ only in the places I expect, where we have that codon
assert  (~(toutput == ntoutput)[tdata[0][:,52,:]==1]).all()
assert (toutput == ntoutput)[tdata[0][:,52,:]==0].all()

#Now do a likelihood profile to see if indeed that parameter is at it's max
val=0.5
def profloss(val,cmodel,tdata=tdata,codonnum=20):
	# print(val)
	#modify our model
	newmodel = copy.deepcopy(cmodel)
	newmodel.conv.weight[:,codonnum,:] += val
	#make the new output
	ntoutput = newmodel(tdata)
	toutput == ntoutput
	#verify they differ only in the places I expect, where we have that codon
	if val!=0: assert  (~(toutput == ntoutput)[tdata[0][:,codonnum,:]==1]).all()
	assert (toutput == ntoutput)[tdata[0][:,codonnum,:]==0].all()
	# return regcriterion(ntoutput,ttarget)
	return nbregcriterion(ntoutput,ttarget)
ls = np.linspace(-1,0,20);
txtplot(ls,np.array([float(profloss(val,cmodel)) for val in ls]))
ls = np.linspace(0,2,20);
txtplot(ls,np.array([float(profloss(val,cmodel)) for val in ls]))



####Okay so... no. it looks like 


