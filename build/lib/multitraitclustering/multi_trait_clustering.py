"""
Author: Hayley Wragg
Created: 10th December 2024
Description:
    This module provides functions for performing multi-trait clustering analysis.
    It includes functions for:
    - Calculating the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)
        for evaluating clustering results.
    - Applying various clustering methods to association data.
    - Generating descriptive strings for clustering methods based on their parameters.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Import local
from multitraitclustering import string_funcs as hp
from multitraitclustering import data_setup as dsetup
from multitraitclustering import clustering_methods as methods

# TODO #4 create a function which loads the data and runs the clustering methods in one call
# TODO #2 save results and plots from clustering

def method_string(meth_str, alg_str, dist_str, num):
    """Create a string describing the clustering method and it's parameters.

    method + algorithm type + distance type + number in title case words"""
    if not isinstance(meth_str, str):
        error_string = "meth_str should be a string not " + str(type(meth_str))
        raise TypeError(error_string)
    if not isinstance(alg_str, str):
        error_string = "alg_str should be a string not " + str(type(alg_str))
        raise TypeError(error_string)
    if not isinstance(dist_str, str):
        error_string = "dist_str should be a string not " + str(type(dist_str))
        raise TypeError(error_string)
    if not isinstance(num, (int, float)):
        error_string = "num should be a number not " + str(type(num))
        raise TypeError(error_string)
    title_str = meth_str.title() + alg_str.title() + dist_str.title()
    return title_str + hp.num_to_word(num)


def get_aic(clust_mem_dist, dimension=2):
    """Calculate the Akaike Information Criterion (AIC)

    Args:
      clust_mem_dist (pd.Dataframe): Distance for point to cluster centroid.
                     rows are data-point, has columns `clust_dist` `clust_num`
      dimension (int): The dimension for each data-point

    .. math::


      k = \text{number of clusters}
      n = \text{number of points}
      sse = \text{variance of the cluster distances}
      \text{log}\_\text{likelihood} = -n / 2 \log(\frac{sse}{n})
      aic = 2 * k * (\text{dimension} + 1) - 2 * \text{log}\_\text{likelihood}

    Returns:

          aic (float): value for the AIC
    """
    if not isinstance(clust_mem_dist, pd.DataFrame):
        error_string = """The input clust_mem_dist to get_aic should be
            of type dataframe not """ + str(type(clust_mem_dist))
        raise TypeError(error_string)
    if not isinstance(dimension, int):
        error_string = """The input dimension to get_aic should be of
            type int not """ + str(type(dimension))
        raise TypeError(error_string)
    if dimension < 2:
        error_string = "dimension should be at least 2 not " + str(dimension)
        raise ValueError(error_string)
    if "clust_num" not in clust_mem_dist.columns:
        error_string = """col clust_num should be in the dataframe and isn't.
            Available columns are, """ + str(clust_mem_dist.columns)
        raise ValueError(error_string)
    if "clust_dist" not in clust_mem_dist.columns:
        error_string = """col clust_dist should be in the dataframe and isn't.
            Available columns are, """ + str(clust_mem_dist.columns)
        raise ValueError(error_string)
    # Number of clusters
    k = clust_mem_dist.clust_num.unique().shape[0]

    # Number of data points
    n = clust_mem_dist.shape[0]

    # Sum of the squared distances
    sse = np.var(clust_mem_dist.clust_dist)

    # Log-likelihood estimation
    log_likelihood = -n / 2 * np.log(sse / n)

    # AIC calculation
    aic = 2 * k * (dimension + 1) - 2 * log_likelihood

    return aic


def get_bic(clust_mem_dist, n_params=-1, dimension=2):
    """Calculate the Akaike Information Criterion (AIC)

    Args:
      clust_mem_dist (pd.Dataframe): Distance for point to cluster centroid.
                        indexed with the data-point label.
                        has terms `clust_dist` `clust_num`
      n_params (int): no. of parameters estimated. If -1 use no. of clusters.
                     Default -1
      dimension (int): dimension for each data-point. Default 2

    .. math::


      k = \text{number of clusters}
      n = \text{number of points}
      sse = \text{variance of the cluster distances}
      \text{log}\_\text{likelihood} = -n / 2 * \log(\frac{sse}{n})
      bic = \log(n)*k*(\text{dimension}+1)-2*\text{log}\_\text{likelihood}

    Returns:

      bic (float): value for BIC
    """
    if not isinstance(clust_mem_dist, pd.DataFrame):
        error_string = """The input clust_mem_dist to get_bic should be
            of type dataframe not """ + str(type(clust_mem_dist))
        raise TypeError(error_string)
    if not isinstance(dimension, int):
        error_string = """The input dimension to get_bic should be of
            type int not """ + str(type(dimension))
        raise TypeError(error_string)
    if not isinstance(n_params, int):
        error_string = """The input n_params to get_bic should be of
            type int not """ + str(type(n_params))
        raise TypeError(error_string)
    if dimension < 2:
        error_string = "dimension should be at least 2 not " + str(dimension)
        raise ValueError(error_string)
    if "clust_num" not in clust_mem_dist.columns:
        error_string = """col clust_num should be in the dataframe and isn't.
            Available columns are, """ + str(clust_mem_dist.columns)
        raise ValueError(error_string)
    if "clust_dist" not in clust_mem_dist.columns:
        error_string = """col clust_dist should be in the dataframe and isn't.
            Available columns are, """ + str(clust_mem_dist.columns)
        raise ValueError(error_string)

    # Number of parameters, if not input then set to the number of clusters
    if n_params == -1:
        k = clust_mem_dist.clust_num.unique().shape[0]
    else:
        k = n_params

    # Number of data points
    n = clust_mem_dist.shape[0]

    # Sum of the squared distances
    sse = np.var(clust_mem_dist.clust_dist)

    # Log-likelihood estimation
    log_likelihood = -n / 2 * np.log(sse / n)

    # BIC calculation
    bic = np.log(n) * k * (dimension + 1) - 2 * log_likelihood

    return bic


def cluster_all_methods(exp_df, assoc_df):
    """Apply all clustering methods to association data and return clusters

    Args:
        exp_df (pd.Dataframe): Association score with the exposure
        assoc_df (pd.Dataframe): Association score with traits. Normalised.

    Returns:
        out_dict (dict): {"clust_pars_dict": dictionary of cluster parameters
                           -keys matching cluster label.,
                          "clust_results": Dataframe with rows corresponding
                          to snps and columns to the cluster methods.}
    Raises:
        TypeError: If exp_df or assoc_df are not pandas DataFrames.
        KeyError: If the 'EXP' column is not present in exp_df.
        ValueError: If the number of rows in assoc_df and exp_df do not match.
        ValueError: If the keys between the clustering method parameter labels
                    and clustering results do not match.
    """
    if not isinstance(exp_df, pd.DataFrame):
        error_string = """Expected dataframe for the exposure scores
            but got """ + str(type(exp_df))
        raise TypeError(error_string)
    if not isinstance(assoc_df, pd.DataFrame):
        error_string = """Expected dataframe for the traits scores
            but got """ + str(type(assoc_df))
        raise TypeError(error_string)
    if "EXP" not in exp_df.columns:
        error_string = f"""There must be a columns named EXP in exp_df.
            Available columns: {exp_df.columns}"""
        raise KeyError(error_string)
    if assoc_df.shape[0] != exp_df.shape[0]:
        error_string = f"""no. of points in assoc_df is {assoc_df.shape[0]} mismatch with
                      {exp_df.shape[0]} in eff_df"""
        raise ValueError(error_string)
    # Initialise results df
    res_df = exp_df.merge(assoc_df,
                          left_index=True,
                          right_index=True,
                          how="inner")

    # Compute cosine-similarity
    beta_mat = res_df.to_numpy()
    cos_sims = cosine_similarity(beta_mat)
    cos_dist = pd.DataFrame(
        index=assoc_df.index,
        data=(1 - cos_sims).clip(min=0),
        columns=assoc_df.index
    )
    euc_dist = pd.DataFrame(
        index=assoc_df.index,
        data=dsetup.mat_dist(res_df),
        columns=assoc_df.index
    )

    clust_dict = {}
    # Cluster with Kmeans - Lloyd - Euclidean - 4 clusters.
    alg = "lloyd"
    dist_met = "Euclidean"
    nclust = 4
    method_str = method_string("Kmeans", alg, dist_met, nclust)
    res_dict = methods.kmeans(
        assoc_df,
        euc_dist,
        res_df,
        nclust=nclust,
        kmeans_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmeans - Lloyd - Euclidean - 6 clusters.
    nclust = 6
    method_str = method_string("Kmeans", alg, dist_met, nclust)
    res_dict = methods.kmeans(
        assoc_df,
        euc_dist,
        res_df,
        nclust=nclust,
        kmeans_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmeans - Elkan - Euclidean - 4 clusters.
    alg = "elkan"
    nclust = 4
    method_str = method_string("Kmeans", alg, dist_met, nclust)
    res_dict = methods.kmeans(
        assoc_df,
        euc_dist,
        res_df,
        nclust=nclust,
        kmeans_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmeans - Elkan - Euclidean - 6 clusters.
    nclust = 6
    method_str = method_string("Kmeans", alg, dist_met, nclust)
    res_dict = methods.kmeans(
        assoc_df,
        euc_dist,
        res_df,
        nclust=nclust,
        kmeans_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmedoids - Euclidean - 4 clusters.
    nclust = 4
    alg = "alternate"
    method_str = method_string("Kmedoids", alg, dist_met, nclust)
    res_dict = methods.kmedoids(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        kmedoids_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmedoids - Euclidean - 6 clusters.
    nclust = 6
    method_str = method_string("Kmedoids", alg, dist_met, nclust)
    res_dict = methods.kmedoids(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        kmedoids_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmeans - Lloyd - CosineSimilarity - 4 clusters.
    nclust = 4
    dist_met = "CosineSimilarity"
    alg = "lloyd"
    method_str = method_string("Kmeans", alg, dist_met, nclust)
    res_dict = methods.kmeans(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        kmeans_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmeans - Lloyd - CosineSimilarity - 6 clusters.
    nclust = 6
    dist_met = "CosineSimilarity"
    alg = "lloyd"
    method_str = method_string("Kmeans", alg, dist_met, nclust)
    res_dict = methods.kmeans(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        kmeans_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmedoids - CosineSimilarity - 4 clusters.
    nclust = 4
    dist_met = "CosineSimilarity"
    alg = "alternate"
    method_str = method_string("Kmedoids", alg, dist_met, nclust)
    res_dict = methods.kmedoids(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        kmedoids_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with Kmedoids - CosineSimilarity - 6 clusters.
    nclust = 6
    dist_met = "CosineSimilarity"
    method_str = method_string("Kmedoids", alg, dist_met, nclust)
    res_dict = methods.kmedoids(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        kmedoids_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with DBSCAN - eps 0.5 - min_s 5
    alg = "auto"
    min_s = 4
    eps = 0.5
    meth_iter_str = "DBSCAN" +  f"{int(eps * 100)}"
    method_str = method_string(meth_iter_str, alg, dist_met, min_s)
    res_dict = methods.dbscan(
        assoc_df,
        cos_dist,
        res_df,
        eps=eps,
        min_s=min_s,
        db_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with DBSCAN - eps 0.6 - min_s 5
    min_s = 4
    eps = 0.6
    meth_iter_str ="DBSCAN" +  f"{int(eps * 100)}"
    method_str = method_string(meth_iter_str, alg, dist_met, min_s)
    res_dict = methods.dbscan(
        assoc_df,
        cos_dist,
        res_df,
        eps=eps,
        min_s=min_s,
        db_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with DBSCAN - eps 0.7 - min_s 5
    alg = "auto"
    min_s = 4
    eps = 0.7
    meth_iter_str = "DBSCAN" +  f"{int(eps * 100)}"
    method_str = method_string(meth_iter_str, alg, dist_met, min_s)
    res_dict = methods.dbscan(
        assoc_df,
        cos_dist,
        res_df,
        eps=eps,
        min_s=min_s,
        db_alg=alg,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with GMM n_comps = 2
    nc = 2
    cov_type = "diag"
    dist_met = "CosineDistance"
    method_str = method_string("GMM", cov_type, dist_met, nc)
    res_dict = methods.gmm(
        assoc_df,
        cos_dist,
        res_df,
        n_comps=nc,
        cov_type=cov_type,
        dist_met=dist_met
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with GMM n_comps = 4
    nc = 4
    method_str = method_string("GMM", cov_type, dist_met, nc)
    res_dict = methods.gmm(
        assoc_df,
        cos_dist,
        res_df,
        n_comps=nc,
        cov_type=cov_type,
        dist_met=dist_met
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with GMM n_comps = 6
    nc = 6
    method_str = method_string("GMM", cov_type, dist_met, nc)
    res_dict = methods.gmm(
        assoc_df,
        cos_dist,
        res_df,
        n_comps=nc,
        cov_type=cov_type,
        dist_met=dist_met
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with birch thresh = 0.25
    thresh = 0.25
    branch_fac = 50
    dist_met = "CosineDistance"
    method_str = method_string(
        "Birch" +  f"{int(thresh * 100)}", "", dist_met, branch_fac
    )
    res_dict = methods.birch(
        assoc_df,
        cos_dist,
        res_df,
        thresh=thresh,
        branch_fac=branch_fac,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with birch thresh = 1.25
    thresh = 1.25
    branch_fac = 50
    dist_met = "CosineDistance"
    method_str = method_string(
        "Birch" +  f"{int(thresh * 100)}", "", dist_met, branch_fac
    )
    res_dict = methods.birch(
        assoc_df,
        cos_dist,
        res_df,
        thresh=thresh,
        branch_fac=branch_fac,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with birch thresh = 2.25
    thresh = 2.25
    branch_fac = 50
    dist_met = "CosineDistance"
    method_str = method_string(
        "Birch" + f"{int(thresh * 100)}", "", dist_met, branch_fac
    )
    res_dict = methods.birch(
        assoc_df,
        cos_dist,
        res_df,
        thresh=thresh,
        branch_fac=branch_fac,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with kmeans mini-batch n= 4
    nclust = 4
    batch_size = 30
    dist_met = "CosineDistance"
    meth_iter_str = "MiniBatchKmeans" + str(batch_size)
    method_str = method_string(meth_iter_str, "", dist_met, nclust)
    res_dict = methods.kmeans_minibatch(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        batch_size=batch_size,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Cluster with kmeans mini-batch n=6
    nclust = 6
    batch_size = 30
    dist_met = "CosineDistance"
    meth_iter_str = "MiniBatchKmeans" + str(batch_size)
    method_str = method_string(meth_iter_str, "", dist_met, nclust)
    res_dict = methods.kmeans_minibatch(
        assoc_df,
        cos_dist,
        res_df,
        nclust=nclust,
        batch_size=batch_size,
        dist_met=dist_met,
    )
    res_df = res_dict["results"]
    clust_dict[method_str] = res_dict["cluster_dict"]

    # Collect results
    clust_res_df = res_df.loc[:, res_df.columns.difference(assoc_df.columns)]
    diff_cols = clust_res_df.columns.difference(exp_df.columns)
    clust_res_df = clust_res_df.loc[:, diff_cols]

    # Differing keys
    diff_a = set(clust_dict.keys()).difference(set(clust_res_df.columns))
    diff_b = set(clust_res_df.columns).difference(set(clust_dict.keys()))
    # Check that the results have matching labels
    if not set(clust_dict.keys()) == set(clust_res_df.columns):
        error_string = f"""keys between the clustering method parameter labels
            and clustering results do not match. Parameter keys not in
            cluster results: {diff_a}, and Results labels not in
            parameter keys: {diff_b}"""
        raise ValueError(error_string)

    out_dict = {"clust_pars_dict": clust_dict, "clust_results": clust_res_df}
    return out_dict
