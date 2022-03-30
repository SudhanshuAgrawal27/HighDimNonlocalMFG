# HighDimNonlocalMFG

Code repository for [*Random Features for High-Dimensional Nonlocal Mean-Field Games*](https://www.sciencedirect.com/science/article/pii/S002199912200198X).

## Set up 
``` 
git clone https://github.com/SudhanshuAgrawal27/HighDimNonlocalMFG
```

There are two experiments - one involving the 8 Gaussian distributions and one involving a bottleneck. 

The set ups of the experiments can be modified in the `eight-gaussians.jl` and `bottleneck.jl` files 

Once the parameters are set, experiments can simply be run using 
```
julia eight-gaussians.jl
```
Plots generated during training are stored in `./figures` and the results are stored in `./data`. 
## Reference

Please cite as  
```
@article{random-features-nonlocal-mfgs,
title = {Random features for high-dimensional nonlocal mean-field games},
journal = {Journal of Computational Physics},
volume = {459},
pages = {111136},
year = {2022},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2022.111136},
url = {https://www.sciencedirect.com/science/article/pii/S002199912200198X},
author = {Sudhanshu Agrawal and Wonjun Lee and Samy {Wu Fung} and Levon Nurbekyan},
keywords = {Mean-field games, Nonlocal interactions, Random features, Optimal control, Hamilton-Jacobi-Bellman},
abstract = {We propose an efficient solution approach for high-dimensional nonlocal mean-field game (MFG) systems based on the Monte Carlo approximation of interaction kernels via random features. We avoid costly space-discretizations of interaction terms in the state-space by passing to the feature-space. This approach allows for a seamless mean-field extension of virtually any single-agent trajectory optimization algorithm. Here, we extend the direct transcription approach in optimal control to the mean-field setting. We demonstrate the efficiency of our method by solving MFG problems in high-dimensional spaces which were previously out of reach for conventional non-deep-learning techniques.}
}
```
## Acknowledgements 
Wonjun Lee, Levon Nurbekyan, and Samy Wu Fung were partially funded by AFOSR MURI FA9550-18-502, ONR N00014-18-1-2527, N00014-18-20-1-2093, and N00014-20-1-2787.


