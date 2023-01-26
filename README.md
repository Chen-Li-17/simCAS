# simCAS
simCAS is an embedding-based method for simulating single-cell chromatin accessibility sequencing data. simCAS is a comprehensive and flexible simulator which provides three simulation modes:  pseudo-cell-type mode, discrete mode and continuous mode. In pseudo-cell-type mode simCAS simulates data with high resemblance to real data. In discrete or continuous mode, simCAS simualtes cells with discrete or continuous states. simCAS also provides functions to simulate data with batch effects and interactive peaks. The synthetic scCAS data generated by simCAS can benifit the benchmarking of various computational methods.
<p align="center">
  <img src="https://github.com/Chen-Li-17/simCAS/blob/main/inst/Fig1-overview.png" width=50%>
</p>

## Installation
```
Requirements:
1. Python 3.8.13 or greater version
2. Packages for simCAS
  numpy (==1.21.0)
  pandas (==1.3.5)
  scipy (==1.4.1)
  rpy2 (==3.5.5)
  sklearn (==1.2.0)
  scanpy (==1.9.1)
  Bio (==1.5.2)
  anndata (==0.8.0)
  statmodels (0.13.2)
3. Packages for tutorials
  matplotlib (==3.5.1)
  seaborn (==0.11.2)
  umap (==0.5.2)

Package installation:
$ git clone https://github.com/Chen-Li-17/simCAS
$ cd simCAS   
$ pip install -r requirements.txt
```

## Tutorial
### Jupyter notebook
For each simulation mode, we provide a detailed tutorial:
- [Pesudo-cell-type mode](https://github.com/Chen-Li-17/simCAS/blob/main/code/tutorial_pseudo-cell-type.ipynb)
- [Discrete mode](https://github.com/Chen-Li-17/simCAS/blob/main/code/tutorial_discrete.ipynb)
- [Continuous mode](https://github.com/Chen-Li-17/simCAS/blob/main/code/tutorial_continuous.ipynb)

### Parameter setting
The simulation function is **simCAS_generate**, the details of the parameters are shown as below:

```
  generate scCAS data with three modes: pseudo-cell-type mode, discrete mode, continuous mode.

  Parameter
  ----------
  peak_mean: 1D numpy array, default=None
      Real peak mean of scCAS data. Used for statistical estimation.
  lib_size: 1D numpy array, default=None
      Real library size of scCAS data. Used for statistical estimation.
  nozero: 1D numpy array, default=None
      Real non-zeros of scCAS data. Used for statistical estimation.
  n_peak: int, default=1e5
      Number of peaks in the synthetic data.
  n_cell_total: int, default=1500
      Number of simulating cells.
  rand_seed: int, default=2022
      Random seed for generation.
  zero_prob: float, default=0.5
      The probability of zeros in PEM.
  zero_set: str, default='all'
      How to set the PEM values to zero.
      1.'all': set the PEM values to zero with probability zero_prob for the whole PEM.
      2.'by_row': set the PEM values to zero with probability zero_prob for each row (peak) of PEM.
  effect_mean: float, default=0.0
      Mean of the Gaussian distribution, from which the PEM values are sampled.
  effect_sd: float, default=0.0
      Standard deviation of the Gaussian distribution, from which the PEM values are sampled.
  min_popsize: int, default=300
      The cell number of the minimal population set in the discrete mode. The number should be less than n_cell_total.
  min_pop: str, default=None
      The name of the minimal population. This should be contained pops_name
  tree_text: str, default=None
      A string of Newick format. In discrete mode this is used to define the covariance matrix of different populations. In continuous mode this is used to provide the differentiation trajectory.
  pops_name: list, default=None
      A list of defined names of populations.
  pops_size: list, default=None
      The number of cells of corresponding populations in the pops_name.
  embed_mean_same: float, default=1.0
      Mean of the Gaussian distritbution, from which the homogeneous CEM values are sampled.
  embed_sd_same: float, default=0.5
      Standard deviation of the Gaussian distritbution, from which the homogeneous CEM values are sampled.
  embed_mean_diff: float, default=1.0
      Mean of the Gaussian distritbution, with which the heterogeneous CEM values are generated.
  embed_sd_diff: float, default=0.5
      Standard deviation of the Gaussian distritbution, with which the heterogeneous CEM values are generated.
  len_cell_embed: int, default=12
      The number of the total embedding dimensions.
  n_embed_diff: int, default=10
      The number of the heterogeneous embedding dimensions.
  simu_type: str, default='cell_type'
      1.'cell_type': Pseudo-cell-type mode. Simulate data resembling real data.
      2.'discrete': Simulate cells with discrete populations.
      3.'continuous': Simulate cells with continuous trajectories.
  correct_iter: int, default=2
      The iterations of correction.
  activation: str, default='exp_linear'
      Choose the activation function to convert parameter matrix values to positive.
      1.'exp_linear': Used in the discrete mode or continuous mode.
      2.'sigmod': Used in the pseudo-cell-type mode.
      3.'exp': Used for biological batch effect generation.
  adata_dir: str, default=None
      The directory of real scCAS anndata.
  distribution: str, default='Poisson'
      Choose the distribution of data.
      1.'Poisson': for count data.
      2.'Bernoulli': for binary data.
  bw_pm: float, default=1e-4
      Band width of the kernel density estimation for peak mean.
  bw_lib: float, default=0.05
      Band width of the kernel density estimation for library size.
  bw_nozero: float, default=0.05
      Band width of the kernel density estimation for non-zero.
  K: float, default=None
      Adjust the slope of the activation function.
  A: float, default=None
      Adjust the slope of the activation function.
  stat_estimation: str, default='one_logser'
      Different discrete distributions to fit peak summation.
      1. 'zero_logser': a variant of logarithmic distribution.
      2. 'one_logser': a variant of logarithmic distribution.
      3. 'ZIP' : zero-inflated Poisson distribution.
      4. 'NB': Negative Binomial distribution.
      4. 'zero_NB': a variant of NB distribution.
      5. 'ZINB': zero-inflated Negative Binomial distribution. 


  Return
  ----------
  adata_final: anndata
      The simulated scCAS data with anndata format. The cell type information is in the observation.
```

