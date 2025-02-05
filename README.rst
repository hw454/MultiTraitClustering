Documentation:
==============
    `https://multitraitclustering.readthedocs.io`

Requirements:
=============
    * python == 3.11.9 (not currently compatible with higher versions)
    * scipy == 1.15.1
    * scikit-learn == 1.6.1
    * scikit-learn-extra == 0.3.0
    * numpy == 2.2.2
    * pandas == 2.2.3

Setup:
=======

..
    TODO get the code-blocks displaying on github!

Installation:
        .. code-block:: unix

            pip install multitraitclustering
Data:
        * Association Data. This is your master data set with association scores corresponding
        to a collection of GWAS analysis. The columns correspond to traits, the rows and snps and the values are the association between the snp and the trait. This should be stored as a `.csv` file with the trait labels on the first row and the snps on the first column. 
        * Exposure Data. This is a dataset corresponding to only the exposure phenotype and the association with the snps. Also stored as a csv file, rows are snps, one column for the exposure labelled `EXP` and the values are the scores.
        * P-values and Standard Error values are not yet incorporated into the analysis.

Usage Example:
================

Loading Data
------------

To load data located in a directory named `data` whose parent directory is your working directory.
The filenames `eff_dfs.csv` and `exp_dfs.csv` can be replaced with alternative filenames to match
your files. The files must be in `csv` format.

.. code-block:: python

    from multitraitclustering import data_setup as ds
    data_sets = ds.load_association_data("./data", 
                                        eff_fname ="/eff_dfs.csv",
                                        exp_fname = "/exp_dfs.csv")

This will give an output in the form of:

.. code-block:: python

    data_sets = {'eff_df':X1687     X1697     X1717        X20015_irnt  X21001_irnt
    rs115866895  0.086834 -0.970112 -1.105207     0.668033     0.402944    
    rs4648450   -0.070006 -0.211097 -1.109551     0.114491    -0.253709  
    rs12024554   0.202266  0.700323 -0.281241     3.877386    -0.650558    
    rs1097327   -0.047729 -0.370089 -0.148402     0.550493    -0.659329  
    rs3737992   -0.741344  1.597846 -1.133015     1.346192    -0.489776   ,
    'exp_df': 	EXP
    rs115866895	-0.013280
    rs4648450	-0.010908
    rs12024554	-0.009474
    rs1097327	-0.009443
    rs3737992	-0.010055 }


Clustering analysis
--------------------

.. code-block:: python

    from multitraitclustering import data_setup as ds
    from multitraitclustering import multi_trait_clustering as mtc

    data_sets = ds.load_association_data("./data", 
                                    eff_fname ="/eff_dfs.csv",
                                    exp_fname = "/exp_dfs.csv")

    cluster_res = mtc.cluster_all_methods(data_sets["eff_df"], data_sets["exp_df"])


The results contains the clusters and the clustering parameters `clust_pars_dict`
and `clust_results`.

.. code-block:: python

    clust_pars = cluster_res["clust_pars_dict"]
    clust_df = cluster_res["clust_results"]

