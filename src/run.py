run -i "src/fake_codon_data.py"
lr = 0.05 # learning rate
epochs = 5 # The number of epochs
bptt=100
run -i  "src/convmodel_ultrasimple.py"

run -i "src/parse_codon_data.py"
lr = 0.05 # learning rate
epochs = 10 # The number of epochs
bptt=300
run -i  "src/convmodel_ultrasimple.py"
