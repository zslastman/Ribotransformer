
from rpy2.robjects import numpy2ri
import rpy2.robjects.lib.ggplot2 as ggplot2

from rpy2.robjects import numpy2ri
numpy2ri.activate()
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
# numpy2ri.deactivate()

def txtplot(a,b=None):
    if b is None:
        b = a
        a = list(range(0,len(b)))
    plx.clear_plot()
    plx.scatter(
            np.array(a),
            np.array(b),
            rows = 17, cols = 70
            )
    plx.show()

from rpy2.robjects import r
rpy2.robjects.globalenv['myv'] = rpy2.robjects.FloatVector([1.,2.])
rpy2.robjects.globalenv['myv2'] = rpy2.robjects.FloatVector([1.,2.])
rpy2.robjects.globalenv['myv'] = np.array([1.,2.])
rpy2.robjects.globalenv['myv2'] = np.array([1.,2.])
r('''
    pdf('tmp.pdf')
    print(qplot(x=myv,y=myv2)+scale_x_continuous())
    dev.off()
    message(normalizePath('tmp.pdf',must=T))
    ''')

ro=rpy2.robjects
from rpy2.robjects.conversion import localconverter

with localconverter(ro.default_converter + pandas2ri.converter):
    pddf = pd.DataFrame({'a':[1,2],'b':[1,2]})
    rpy2.robjects.globalenv['mydf'] = pddf
    
ro.globalenv['rsignal'] =  rdata.ribosignal.sum(axis=0).numpy()
r('library(txtplot);txtplot(rsignal)')

  # r_from_pd_df = ro.conversion.py2rpy(pd_df)

txtplot(rdata.ribosignal[0,0:20])



with localconverter(ro.default_converter + pandas2ri.converter):
    rpy2.robjects.globalenv['mydf'] = dtcompdf
r('''library(txtplot);library(tidyverse);
    txtplot(mydf[[3]],log(mydf[[4]]));
   cor(mydf[[3]],log(mydf[[4]]))
''')
