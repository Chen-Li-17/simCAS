# simCAS
simCAS is an embedding-based method for simulating single-cell chromatin accessibility sequencing data. simCAS is a comprehensive and flexible simulator which provides three simulation modes:  pseudo-cell-type mode, discrete mode and continuous mode. In pseudo-cell-type mode simCAS simulates data with high resemblance to real data. In discrete or continuous mode, simCAS simualtes cells with discrete or continuous states. simCAS also provides functions to simulate data with batch effects and interactive peaks. The synthetic scCAS data generated by simCAS can benifit the benchmarking of various computational methods.
![alt text](https://github.com/Chen-Li-17/simCAS/blob/main/inst/Fig1-overview.png)

## Installation
```
Requirements:
1. Python 3.8.13 or greater version
2. Packages for simCAS
- numpy (==1.21.0)
- pandas (==1.3.5)
- scipy (==1.4.1)
- rpy2 (==3.5.5)
- sklearn (==1.2.0)
- scanpy (==1.9.1)
- Bio (==1.5.2)
- anndata (==0.8.0)
- statmodels (0.13.2)
3. Packages for tutorials
- matplotlib (==3.5.1)
- seaborn (==0.11.2)
- umap (==0.5.2)

```
