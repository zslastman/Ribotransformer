library(tidyverse)



#actual ix _tr
trixlossdata = read_tsv('../Ribotransformer/norm_trans_losses.tsv')
#
trixlossdata = read_tsv('../Ribotransformer/ix_tr_norm_trans_losses.tsv')



plotfile <- here(paste0('plots/','tr_losscurves','.pdf'))
pdf(plotfile)
trixlossdata%>%
	group_by(epoch)%>%mutate(epoch = epoch + seq(0,1,len=n()))%>%
	pivot_longer(training:validation,names_to='curve',values_to='values')%>%
	ggplot(aes(y=values,x=epoch,color=curve))+
	geom_point()+
	geom_smooth()+
	scale_x_continuous('epoch')+
	scale_y_continuous('Rho')+
	geom_hline(yintercept=0.45,linetype='dashed')+
	#facet_grid()+
	theme_bw()
dev.off()
message(normalizePath(plotfile))



profdat = read_csv('../Ribotransformer/gprofdat.csv')[[2]]
profpred = read_csv('../Ribotransformer/gprofdatpred.csv')[[2]]
profdat = profdat - mean(profdat[profdat!=0])
profdat = profdat / sd(profdat[profdat!=0])
profpred[profpred==-4]=NA
plotfile <- here(paste0('plots/','HASPIN_profile','.pdf'))
pdf(plotfile)
tibble(data=profdat,pred=profpred,pos=seq_along(profdat))%>%
	pivot_longer(data:pred,names_to='type')%>%
	ggplot(aes(x=pos,y=value))+
	facet_grid(type~.,scale='free_y')+
	geom_point()+
	scale_x_continuous()+
	scale_y_continuous()+
	theme_bw()
dev.off()
message(normalizePath(plotfile))


################################################################################
########## Time barplot for orfquant vs ribostan
################################################################################
plotfile <- here(paste0('plots/','timingplot_5.9Mrds','.pdf'))
pdf(plotfile,w=4,h=4)
tibble(time = c(27.3/60,262/60),program=c('RiboStan','ORFquant')%>%as_factor)%>%
	ggplot(aes(y=time,x=program))+
	stat_identity(geom='bar')+
	ylab('time (hours)')
	scale_y_continuous()+
	theme_bw()
dev.off()
message(normalizePath(plotfile))

