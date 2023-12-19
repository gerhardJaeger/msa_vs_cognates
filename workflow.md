# Workflow

All steps are executed in a linux environment as defined in `Dockerfile`.

- create character matrix files

  ```{bash}
  > cd code
  > python get_nexus_from_cognate_classes.py
  > python get_nexus_from_correspondences.py  
  > python get_phylip_from_cognate_classes.py  
  > python get_phylip_from_correspondences.py
  ```

- create goldstandard trees from Glottolog

  ```{shell}
  > python get_glottolog_trees.py
  ```

- create *MrBayes* scripts and combined nexus files

  ```{shell}
  > cd code
  > julia create_mb_scripts.jl
  ```

- run *MrBayes* scripts (to be run on a server with at least 100 cores)

  ```{shell}
  > cd mrbayes
  > bash run_mrbayes.sh
  ```

- check convergence of *MrBayes* runs

  ```{shell}
  > cd ..
  > julia check_convergence.jl
  ```

- extract posterior tree samples

  ```{shell}
  > Rscript create_posterior_samples.r
  ```

- compute GQD for Bayesian trees

  ```{shell}
  > julia get_qdists_mb.jl
  ```

- extract $\alpha$-values for Bayesian analysis

  ```
  julia evaluate_alpha.jl
  ```

- run maximum likelihood experiment (script includes execution of RAxML-NG, GQD computation and extraction of $\alpha$-values)

  ```
  cd ml/
  python ml_experiment.py
  ```

  
