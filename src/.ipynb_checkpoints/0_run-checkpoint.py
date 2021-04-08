run -i "src/fake_codon_data.py"
lr = 0.05 # learning rate
epochs = 5 # The number of epochs
bptt=100
run -i  "src/convmodel_ultrasimple.py"


#this works on yeast data
# %run -i "Parse/parse_codon_data_old.py"
lr = 0.01 # learning rate
epochs = 60 # The number of epochs
bptt = 300
# %run -i  "Model/convmodel_ultrasimple_old.py"



#this works on yeast data
run -i "src/Parse/parse_codon_newer.py"
lr = 0.01 # learning rate
epochs = 40 # The number of epochs
bptt = 300
run -i  "src/Model/convmodel_ultrasimple_old.py"


#this also works on yeast data
#on liu....  no
USE_SEQ_FEATURES=False
GET_ESM_TOKENS=False
USE_ESM_TOKENS=False
run -i "Parse/parse_codon_data.py"
lr = 0.01 # learning rate
epochs = 30 # The number of epochs
bptt = 300
run -i  "Model/convmodel_ultrasimple_old.py"



#just codons
USE_SEQ_FEATURES=False
GET_ESM_TOKENS=True
USE_ESM_TOKENS=False
run -i "src/Parse/parse_codon_data.py"
lr = 0.01 # learning rate
epochs = 30 # The number of epochs
bptt = 150
run -i  "src/Model/convmodel_ultrasimple_old.py"

#with tokens
USE_SEQ_FEATURES=False
GET_ESM_TOKENS=True


USE_ESM_TOKENS=False
lr = 0.05 # learning rate
epochs = 20 # The number of epochs
bptt = 300
run -i  "src/Model/convmodel_ultrasimple_old.py"

USE_ESM_TOKENS=True
lr = 0.05 # learning rate
epochs = 20 # The number of epochs
bptt = 300
run -i  "src/Model/convmodel_ultrasimple_old.py"

