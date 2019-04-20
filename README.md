# Lasagna

## Installation
We run all experiments on Python 3.7.1. To install the necessary modules for running our code, run
```
pip install -r requirements.txt
```

## Directory Structure
All files can be downloaded from the [im2recipe site](http://im2recipe.csail.mit.edu/dataset/download/). Our code expects a directory structure as follows:

```
data/
    lmdbs/
        train_keys.pkl
        train_lmdb
        val_keys.pkl
        val_lmdb
        test_keys.pkl
        test_lmdb
    images/
        train/
        val/
        test/
```

## Running Experiments
We specify our script to accept parameters from the command line. For more information, please view the documentation in `train_calorie.py`. To run our experiments for our best performing model on classification, execute the following on `bash`:

```
python3.7 train_calorie.py --batch_size 128 --pos_weight 10 --model_name resnet50
```

This requires approximately 11 GB of GPU memory, and takes approximately 12 hours to finish training.
