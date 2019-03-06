### CS7641 Assignment 3
### Keun-Hwi Lee(klee754)
### Fall2018

This project contains all of the necessary code
and data to run the experiments used in CS7641 Assignment 3.

## Code
https://github.com/QuinnLee/cs-7641-A3

## Setup
1. Ensure you have Python 2.7 installed (any version above 2.7.10 will work) on your system, along with all standard libraries including the 'pip' and 'setuptools' packages.

2. Run 'pip install -r requirements.txt' to install the necessary Python libraries.

3. Optional Also have `juypter` installed for NN experiments

## Running the experiments

1. To perform data pre-processing run the main() functions in the 'data_preprocess.py.' The processed data sets will be written to the 'data/experiments' folder.

2. To run all the experiments and produce all results CSVs/plots
simply run the `run.sh` script in the "src" folder.
It will execute the various Python modules in the correct order and output all the relevant results.
The chosen K values for clusters and `dims` for dimension reduction algorithms are hard coded

2.1 For clustering `clustering.py --generate 1` runs the experiment, `clustering.py` creates validation plots and contingency plots after the given `k` values are hardcoded in
2.2 The `--dimension 1` flag runs the dimension reduction experiment on the given file
2.3 The `--cluster_exp 1` flag runs the cluster experiment on the reduced data, the `# component` values are hard coded
2.4 No flag creates the cluster validation plots

3. The `NN` experiments are on `NN Experiments.ipynb`. This can be ran doing `jupyter notebook` in the `src` dir. OR run `NN_Experiments.py`

## Credits and References
All implementations and experiment code for this assignment were taken from the
Python scikit-learn library (http://scikit-learn.org)
the experiment code was adapted from Jonathan Tay's repository (https://github.com/JonathanTay/CS-7641-assignment-3) and Federico Ciner (https://github.com/FedericoCiner).
