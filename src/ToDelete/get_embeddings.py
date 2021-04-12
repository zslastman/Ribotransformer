#experiment results. 
torch.load('examples/twoex/0|beta-lactamase_P20P|1.581033423.pt')['representation
    ...: s'][1].shape


#I want to call like this:
#python extract.py esm1_t34_670M_UR50S examples/twoex.fa examples/twoex/     --repr_layers 34 --include per_tok
#This gets me a list of .pt objects I can load with torch.load. The repr layers doesn't seem to make any difference - I always get 286 features this way.