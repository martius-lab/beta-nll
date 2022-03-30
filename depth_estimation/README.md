# Depth Estimation with AdaBins

This folder contains the code for the depth estimation experiments. 
It was adapted from the official repository of [AdaBins](https://arxiv.org/abs/2011.14141): https://github.com/shariqfarooq123/AdaBins by Shariq Farooq Bhat, licensed under GPL v3 license.

We can not give any support for this model. Please post your questions to the original authors [here](https://github.com/shariqfarooq123/AdaBins).

## Training

You can train a model like so:

```
python train.py args_train_nyu_nll.txt
```

## Dataset

To prepare the NYUv2 dataset, follow the instructions here: https://github.com/cleinc/bts. 
The configuration assumes it is placed in a folder `./data/NYUv2`.