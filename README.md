# DistUCB

Official implementation for the paper [More Benefits of Being Distributional: Second-Order Bounds for Reinforcement Learning](https://www.arxiv.org/abs/2402.07198). 

We provide code for regcb and the distUCB algorithm and toggle between the two using the `alg` hyperparameter. The hyperparameters are found in `conf` with the main being `config.yaml`.

## Installation

One can install the required packages by running `pip install -r requirements.txt`.

## Datasets

Both the prudential and housing datasets require preprocessing. To do this, one can run `prudential_preprocessing.py` and `king_housing_preprocessing.py` respectively. To download the prudential dataset, one should look at the Prudential Kaggle competition and use that CSV (https://www.kaggle.com/c/prudential-life-insurance-assessment). Preprocessing this dataset takes a bit of time. The king housing arff dataset is provided under `datasets/dataset.arff`

## Running the code

To run the code, one can use the following command:

```bash
accelerate launch main.py
```
