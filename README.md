# Semi-Supervised Tile2Vec
This github contains the code for the extension of the Tile2Vec paper by Ermon's lab: Jean et al (citation on the bottom). 
Please contact agupta21@stanford.edu and shailk@stanford.edu for the paper.

The primary extensions are as follows:

In examples.py, the substantive notebooks added were 

Semi-supervised Training + Classification Pipeline (Strat 1).ipynb	
Strat 1- Hypertuned.ipynb	
Strat 2 of Semi-Supervised Learning (Original).ipynb	
Strat 2 of Semi-Supervised Learning Copy 1 (HyperP).ipynb	
Visualization of Strat 1.ipynb
Hyperparameter tuning.ipynb

These contain most of the code used for visualizations, running the models, augmenting the dataset based off of the new strategies, pre-processing, etc.

In terms of modification of the underlying models/files, substantive edits were made to:


The loss functions were changed as well as some of the sampling methods in these files. There were also modifications made to enable hypertuning:

tilenet.py
training.py
training_tuned.py

Below are the instructions per the Ermon lab as to how to get the underlying data for training and test.
## Getting data

Follow the follow instructions for getting the necessary data for the provided examples.

### Getting pre-trained NAIP model

To get a Tile2Vec model that has been pre-trained on the NAIP dataset, download the weights provided [here](https://www.dropbox.com/s/bvzriiqlcof5lol/naip_trained.ckpt?dl=0) and save the file in the models directory: `tile2vec/models/naip_trained.ckpt`.

### Getting tiles
Unzip `data/tiles.zip` in the data directory, i.e., `tile2vec/data/tiles`. Within this folder, you should find 1000 tiles saved as Numpy arrays named `{1-1000}tile.npy` and 1 file named `y.npy` that contains the CDL labels corresponding to these 1000 tiles. 

### Getting triplets
Download tile triplet dataset for the training example [here](https://www.dropbox.com/s/afw3cbvo7sjerru/triplets.zip?dl=0) and then unzip in the data directory, i.e., `tile2vec/data/triplets`. This directory contains 100k tile triplets, with each triplet identified by its index and "anchor", "neighbor", or "distant".

Citation:
Jean, Neal et al. “Tile2Vec: Unsupervised representation learning for spatially distributed data.” CoRR abs/1805.02855 (2019).
