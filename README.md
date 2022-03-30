# Code for "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"

This repository contains the code release for the paper ["On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"](https://arxiv.org/abs/2203.09168), published at [ICLR 2022](https://openreview.net/forum?id=aPOpXlnV1T).

This work was done by Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, Georg Martius at the [Autonomous Learning Group](https://al.is.tuebingen.mpg.de/), Max-Planck Institute for Intelligent Systems in Tübingen.

If you make use of our work, please use the citation information [below](#citation).

## Abstract

Capturing aleatoric uncertainty is a critical part of many machine learning systems. In deep learning, a common approach to this end is to train a neural network to estimate the parameters of a heteroscedastic Gaussian distribution by maximizing the logarithm of the likelihood function under the observed data. In this work, we examine this approach and identify potential hazards associated with the use of log-likelihood in conjunction with gradient-based optimizers. First, we present a synthetic example illustrating how this approach can lead to very poor but stable parameter estimates. Second, we identify the culprit to be the log-likelihood loss, along with certain conditions that exacerbate the issue. Third, we present an alternative formulation, termed β-NLL, in which each data point's contribution to the loss is weighted by the β-exponentiated variance estimate. We show that using an appropriate β largely mitigates the issue in our illustrative example. Fourth, we evaluate this approach on a range of domains and tasks and show that it achieves considerable improvements and performs more robustly concerning hyperparameters, both in predictive RMSE and log-likelihood criteria. 

## Quick Implementation of beta-NLL

Here is a ready-to-use snippet for our beta-NLL loss such that you don't have to go dig through the code:

```Python
def beta_nll_loss(mean, variance, target, beta=0.5):
    """Compute beta-NLL loss
    
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    
    return loss.sum(axis=-1)
```

## Setup

We use conda to manage the environment. Use the following command to recreate the environment:

```
conda env create -f environment.yml
```

We only use standard packages: pytorch, torchvision, numpy, scipy, sklearn, pandas, matplotlib.
Activate the environment with `conda activate prob-nns`.

The datasets must reside in the folder `data/` by default.

### Datasets

- UCI datasets are already included in `data/UCI_Datasets`
- MNIST and Fashion-MNIST are handled by torchvision
- ObjectSlide and FPP coming soon

## Experiments

Use the following commands to reproduce the experiments. 

```
./train.sh <dataset> <method> <beta>
```

where `<dataset>` can be one of
- `Sine` (sinusoidal without heteroscedastic noise)
- `SineDetlefsen` (sinusoidal with heteroscedastic noise)
- `ObjectSlide` (dynamics models)
- `FPP` (dynamics models)
- `MNIST` (generative modeling)
- `Fashion-MNIST` (generative modeling)

and `<method>` can be one of `likelihood`, `mse`, `moment_matching`. 
For `likelihood`, you can also specify the value for `<beta>` (default 0).

To reproduce the UCI results, use

```
./train_uci.sh <dataset> <method> <beta>
```

where `<dataset>` can be one of `carbon,concrete,housing,energy,kin8m,naval,power,protein,superconductivity,wine-red,wine-white,yacht`.

The results will be put into the folder `./logs` by default.

For the depth estimation experiments, see [the depth_estimation folder](depth_estimation/).

## Citation

Please use the following citation if you make use of our work:

```
@inproceedings{Seitzer2022PitfallsOfUncertainty,
  title = {On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks},
  author = {Seitzer, Maximilian and Tavakoli, Arash and Antic, Dimitrije and Martius, Georg},
  booktitle = {International Conference on Learning Representations},
  month = apr,
  year = {2022},
  url = {https://openreview.net/forum?id=aPOpXlnV1T},
  month_numeric = {4}
}
```

## License and Attribution

This implementation is licensed under the MIT license.

The dynamics dataset ObjectSlide originally stems from our previous work [Causal Influence Detection for Improving Efficiency in Reinforcement Learning](https://arxiv.org/abs/2106.03443), which you can find here: https://github.com/martius-lab/cid-in-rl.

Code for loading UCI datasets adapted from https://github.com/yaringal/DropoutUncertaintyExps/ by Yarin Gal under CC BY-NC 4.0 license.

Code for [variational variance](https://arxiv.org/abs/2006.04910) and for downloading UCI datasets adapted from https://github.com/astirn/variational-variance by Andrew Stirn under MIT license.

Code for depth estimation with [AdaBins](https://arxiv.org/abs/2011.14141) was adapted from https://github.com/shariqfarooq123/AdaBins by Shariq Farooq Bhat, licensed under GPL v3 license.
