# Genetic engineering attribution with deep learning.

## Quick Start
To reproduce all figures, first clone the repository. Then request the data.

Next, download the conda environment manager, which will come with the needed python 3.6 distro. This has only been tested with Linux, specifically Ubuntu 15.04 and 18.04.

`wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
`bash Miniconda3-latest-Linux-x86_64.sh`
Restart the currrent shell, then create and activate the provided environment. You may want to change the prefix at the bottom of the .yml page. For more info on installing a conda env from a file, see the conda docs (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment).
Install the yml:

`conda env create -f attrib.yml`

You are ready to go! Each notebook in `figures` folder reproduces a figure subpanel, except where stated otherwise.

## Training and Analysis
The `other` folder contains code necessary to reproduce all training and and analysis outside of the figures. It uses two environments, one is the same as the above used for the figures (and found in `attrib.yml`) and the other described below, for use in training deep learning models on GPUs. The latter environment will be referred to as `pytorch_training`. In addition to the escription below, each file should flag the environment in the header.

It's contents are as follows:
- `CNN` Code to train and inference with the CNN architecture modeled off Nielsen and Voigt (2018). Env: `pytorch_training`
- `blast` Configure a blast database and search it for matches to produce the blast baseline. Note this requires the blast command line tool installed (https://www.ncbi.nlm.nih.gov/books/NBK279690/). Env: `attrib`
- `bpe` Learn an enocding of the training set sequence using byte pair encoding and unigram encoding. Env: `pytorch_training`
- `calibration` Learn the temperature scaling parameter for recalibrating the deteRNNt model. Env: `pytorch_training`
- `countries` Train a random forest to predict nation of origin. Env: `attrib`
- `deteRNNt` Training and inference with the  deteRNNt model. Training proceeds in steps noted in the file names. Env: `pytorch_training`
- `lineages` Train a random forest to predict ancestor lab. Env: `attrib`
- `lineages_and_split` Train test split and parse the lineages. Env: `attrib`
- `score` Compute the accuracies, top10 accuracies, and calibration curves for all models. Also produces Figure 2. Env: `attrib`

## pytorch_training environment
Most of the code flagged to be used with the `pytorch_training` environment was run on Amazon Web Services (AWS). This README will assume AWS acccess; please feel free to open an issue if you need help recreating the training environment on another machine or cloud service.

### Environment setup
1. Launch an AWS instance. Most of the training can be accomplished on p2.xlarge instances, and some code (eg `calibration` and `bpe`) do not need GPUs
2. Select the Deep Learning AMI for Ubuntu (this analysis used AMI ami-0f9e8c4a1305ecd22)
3. Connect to the instance and download the data and code as described above.
4. `source activate pytorch_p36` to start the conda environment
5. `pip install -U ray && pip install -U ray[debug]` install ray
6. `pip install sentencepiece` Install sentencepiece for BPE tasks
7. `conda install -y pandas=0.24.1` Ensure that the pandas version can read the picked dataframes.
8. Run the desired code!

