# HighDimNonlocalMFG

Code repository for [*Random Features for High-Dimensional Nonlocal Mean-Field Games*](https://www.sciencedirect.com/science/article/pii/S002199912200198X?dgcid=coauthor).

You can find an explanatory [*blog-post*](https://medium.com/@sudhanshuagrawal2001/100-dimensional-games-de3eb78b1e05) on this paper published on Medium.

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
@article{randomfeaturesmfgs,
title = {Random features for high-dimensional nonlocal mean-field games},
journal = {Journal of Computational Physics},
volume = {459},
pages = {111136},
year = {2022},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2022.111136},
url = {https://www.sciencedirect.com/science/article/pii/S002199912200198X},
author = {Sudhanshu Agrawal and Wonjun Lee and Samy {Wu Fung} and Levon Nurbekyan}
}
```
## Acknowledgements 
Wonjun Lee, Levon Nurbekyan, and Samy Wu Fung were partially funded by AFOSR MURI FA9550-18-502, ONR N00014-18-1-2527, N00014-18-20-1-2093, and N00014-20-1-2787.


