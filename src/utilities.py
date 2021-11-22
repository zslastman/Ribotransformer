from rpy2.robjects import numpy2ri
import rpy2.robjects.lib.ggplot2 as ggplot2

from rpy2.robjects import numpy2ri
numpy2ri.activate()
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import torch
import pandas as pd
import numpy as np
ten = torch.Tensor
# numpy2ri.deactivate()

def numconv(a):
    if 'detach' in dir(a): 
        a = a.detach().numpy().flatten()
    if type(a) is torch.Tensor:
        a=a.numpy().flatten()
    if type(a) is pd.Series:
        a=np.array(a)
    return(a)


def txtplot(a,b=None):
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    import rpy2
    ro = rpy2.robjects
    a = numconv(a)
    if b is None:
        b = a
        a = np.array(list(range(1,1+len(b))))
    b = numconv(b)
    ro.globalenv['a'] =  a
    ro.globalenv['b'] =  b
    ro.r('library(txtplot);txtplot(a,b)')


def cortest(a,b=None):
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    import rpy2
    ro = rpy2.robjects
    a = numconv(a)
    if b is None:
        b = a
        a = np.array(list(range(1,1+len(b))))
    b = numconv(b)
    ro.globalenv['a'] =  a
    ro.globalenv['b'] =  b
    ro.r('cor.test(a,b)')

# txtplot(a=codstrengths,b=testoutput.exp()[:,1:])

a=ten([10])
b=None
def txtdensity(a):
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    import rpy2
    ro = rpy2.robjects
    a = numconv(a)
    ro.globalenv['a'] =  a
    ro.r('library(txtplot);txtdensity(a)')

if False:
    # from rpy2.robjects import r
    # rpy2.robjects.globalenv['myv'] = rpy2.robjects.FloatVector([1.,2.])
    # rpy2.robjects.globalenv['myv2'] = rpy2.robjects.FloatVector([1.,2.])
    # rpy2.robjects.globalenv['myv'] = np.array([1.,2.])
    # rpy2.robjects.globalenv['myv2'] = np.array([1.,2.])
    # r('''
    #     pdf('tmp.pdf')
    #     print(qplot(x=myv,y=myv2)+scale_x_continuous())
    #     dev.off()
    #     message(normalizePath('tmp.pdf',must=T))
    #     ''')

    import rpy2
    ro=rpy2.robjects
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + pandas2ri.converter):
        pddf = pd.DataFrame({'a':[1,2],'b':[1,2]})
        rpy2.robjects.globalenv['mydf'] = pddf
        
    ro.globalenv['rsignal'] =  rdata.ribosignal.sum(axis=0).numpy()
    r('library(txtplot);txtplot(rsignal)')

      # r_from_pd_df = ro.conversion.py2rpy(pd_df)

    ro.globalenv['rsignal'] =  np.log(rdata.ribosignal[0].numpy()+1)
    r('library(txtplot);txtdensity(rsignal)')

    i=0

    statlist = []
    for i in range(0,1000):
        genecounts = rdata.ribosignal[i][rdata.data['codons'][i,0]!=1][10:][:-10].numpy()
        if(np.mean(genecounts==0)<0.05):
            genecounts = np.log(genecounts+1)
            gmean = genecounts.mean()
            gstd = np.std(genecounts)
            statlist.append((gmean,gstd))
            print('gene:'+str(i)+' log mean:'+str(gmean)+' stddev:'+str(gstd))

    ro.globalenv['means'] =  np.array([i[0] for i  in statlist])
    ro.globalenv['stds'] =  np.array([i[1] for i in statlist])
    r('library(txtplot);txtplot(means,stds)')
    r('library(txtplot);txtdensity(means)')
    r('library(txtplot);txtdensity(stds)')

    a=ten([10])
    b=None

    txtplot(a=cmodel.conv.weight)
    txtdensity(cmodel.conv.weight)
    # ro.globalenv['rsignal'] =  genecounts
    # r('library(txtplot);txtdensity(rsignal)')



    with localconverter(ro.default_converter + pandas2ri.converter):
        rpy2.robjects.globalenv['mydf'] = dtcompdf
    r('''library(txtplot);library(tidyverse);
        txtplot(mydf[[3]],log(mydf[[4]]));
       cor(mydf[[3]],log(mydf[[4]]))
    ''')




# import numpy as np
# import matplotlib.pyplot as plt

# # Make some fake data.
# a = b = np.arange(0, 3, .02)
# c = np.exp(a)
# d = c[::-1]

# # Create plots with pre-defined labels.
# fig, ax = plt.subplots()
# ax.plot(a, c, 'k--', label='Model length')
# ax.plot(a, d, 'k:', label='Data length')
# ax.plot(a, c + d, 'k', label='Total message length')

# legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# # Put a nicer background color on the legend.
# legend.get_frame().set_facecolor('C0')

# plt.savefig('tmp.png')
