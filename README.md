# HighDimNonlocalMFG

Code repository for [*Random Features for High-Dimensional Nonlocal Mean-Field Games*](https://arxiv.org/abs/2202.12529).

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
@misc{randomfeaturesMFG2022,
      title={Random Features for High-Dimensional Nonlocal Mean-Field Games}, 
      author={Sudhanshu Agrawal and Wonjun Lee and Samy Wu Fung and Levon Nurbekyan},
      year={2022},
      eprint={2202.12529},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}

```
## Acknowledgements 
Wonjun Lee, Levon Nurbekyan, and Samy Wu Fung were partially funded by AFOSR MURI FA9550-18-502, ONR N00014-18-1-2527, N00014-18-20-1-2093, and N00014-20-1-2787.


