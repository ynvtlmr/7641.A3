#! /bin/bash

python clustering.py --generate 1
python clustering.py

python pca.py --dimension 1
python pca.py --cluster_exp 1
python pca.py

python ica.py --dimension 1
python ica.py --cluster_exp 1
python ica.py

python rp.py --dimension 1
python rp.py --cluster_exp 1
python rp.py

python rf.py --dimension 1
python rf.py --cluster_exp 1
python rf.py

python NN_Experiments.py > ../results/NN/NN_Output.txt
