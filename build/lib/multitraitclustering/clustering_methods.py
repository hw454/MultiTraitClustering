import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, Birch, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture

from multitraitclustering import multi_trait_clustering as mtc
from multitraitclustering import data_processing as dp
from multitraitclustering import checks as checks


def kmeans(
    assoc_df,
    dist_df,
    res_df,
    nclust=4,
    rand_st=240,
    n_in=50,
    init_km="k-means++",
    iter_max=300,
    kmeans_alg="lloyd",
    dist_met="Euclidean",
):
    """Compute the clusters found using the K-means algorithm

    Args:
        assoc_df (pd.Dataframe): Association with the exposure
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe - association data and cluster results
        nclust (int, optional): Number of desired clusters. Defaults to 4.
        rand_st (int, optional): Random number initialisation. Defaults to 240.
        n_in (int, optional): _description_. Defaults to 50.
        init_km (str, optional): Method for initialising the clusters.
                                Defaults to "k-means++".
        iter_max (int, optional): Max no. iterations if no cluster convergence.
                                Defaults to 300.
        kmeans_alg (str, optional): Implementation algorithm. Default "lloyd".
        dist_met (str, optional): Distance metric, "Euclidean" (default),
                                or "CosineSimilarity"

    Raises:
        TypeError: assoc_df is not a dataframe
        TypeError: dist_df is not a dataframe
        TypeError: res_df is not a dataframe
        TypeError: nclust is not an integer
        TypeError: rand_st is not an integer
        TypeError: init_km is not an integer
        TypeError: init_km is not a string
        TypeError: iter_max is not an integer
        TypeError: kmeans_alg is not a string
        TypeError: dist_met is not a string
        ValueError: n_in is not an integer or auto
        ValueError: init_km is not one of: `k-means++`, `random` or an array
        ValueError: kmeans_alg is not one of: “lloyd”, “elkan”
        ValueError: dist_met is not "Euclidean" or "CosineSimilarity"

    Returns:
        Dictionary containing:
        * "results"- results dataframe with appended new clusters
        * "cluster_dict" - dictionary of parameters for clustering iteration
    """
    euc_str = "Euclidean"
    cos_str = "CosineSimilarity"
    # TYPE CHECKS
    # Raise TypeError if assoc_df, dist_df, or res_df are not Dataframes
    checks.df_check(assoc_df, "exposure data assoc_df")
    checks.df_check(dist_df, "distance data dist_df")
    checks.df_check(res_df, "results data res_df")
    # Raise TypeError if n_clust, rand_st, iter_max or n_in are not integers
    checks.int_check(nclust, "nclust")
    checks.int_check(rand_st, "rand_st")
    checks.int_check(iter_max, "iter_max")
    # Raise TyperError if n_in is not an integer or "auto"
    if not isinstance(n_in, (int, str)):
        error_string = """The input n_in should be integer or string not """\
            + str(type(n_in))
        raise TypeError(error_string)
    # Raise TypeError if init_km, kmeans_alg is not a string or an array
    if not isinstance(init_km, (str, np.ndarray)):
        error_string = """The input init_km should be a string not """\
            + str(type(init_km))
        raise TypeError(error_string)
    # Raise TypeError if kmeans_alg is not a string
    checks.str_check(kmeans_alg, "kmeans_alg")
    # VALUE CHECKS
    # Raise ValueError if n_in is not an integer or "auto".
    if isinstance(n_in, str):
        if n_in != "auto":
            error_string = """The input dist_df should be either a integer
            or auto not """ + n_in
            raise ValueError(error_string)
    # Raise ValueError if init_km is not one of: `k-means++`, `random` or array
    if isinstance(init_km, str):
        if init_km not in ["k-means++", "random"]:
            error_string = """The input init_km should be either `k-means++`,
            `random` or an array""" + init_km
            raise ValueError(error_string)
    # Raise ValueError if kmeans_alg is not one of: “lloyd”, “elkan”
    if kmeans_alg not in ["lloyd", "elkan"]:
        error_string = """The input kmeans_alg should be either `lloyd`,
            `elkan` not """ + kmeans_alg
        raise ValueError(error_string)
    # Raise ValueError if dist_met not one of "CosineSimilarity" or "Euclidean"
    if dist_met not in [euc_str, cos_str]:
        error_string = """The input dist_met should be either
                        {euc}, {cos} not {dist}""".format(
            euc=euc_str, cos=cos_str, dist=dist_met
        )
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d
        should match the number of rows in the association data %d""" % (
            res_df.shape[0],
            assoc_df.shape[0],
        )
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match
        the number of columns %d in dist_df""" % (
            dist_df.shape[0],
            dist_df.shape[1],
        )
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association
            data %d does not match the number of rows in the distance
            data %d""" % (assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)

    klab = mtc.method_string("Kmeans", kmeans_alg, dist_met, nclust)
    if dist_met == euc_str:
        n_kmeans = KMeans(
            n_clusters=nclust,
            init=init_km,
            random_state=rand_st,
            n_init=n_in,
            max_iter=iter_max,
            algorithm=kmeans_alg,
        ).fit(assoc_df.to_numpy())
        res_df[klab] = n_kmeans.labels_
        # Centroid distances
        centroids = pd.DataFrame(n_kmeans.cluster_centers_,
                                 columns=assoc_df.columns)
        dist_df = dp.centroid_distance(
            cents=centroids, data=assoc_df, membership=res_df[klab]
        )
    elif dist_met == "CosineSimilarity":
        n_kmeans = KMeans(
            n_clusters=nclust,
            init=init_km,
            random_state=rand_st,
            n_init=n_in,
            max_iter=iter_max,
            algorithm=kmeans_alg,
        ).fit(dist_df.to_numpy())
        res_df[klab] = n_kmeans.labels_
        # Centroid distances
        centroids = pd.DataFrame(n_kmeans.cluster_centers_)
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=dist_df,
            membership=res_df[klab],
            metric="cosine-sim"
        )
    # Calculate the AIC
    dimension = len(assoc_df.columns)
    km_aic = mtc.get_aic(dist_df, dimension)
    km_bic = mtc.get_bic(dist_df, dimension=dimension)
    cluster_dict = {
        "alg_name": "K-means",
        "nclust": nclust,
        "rand_st": rand_st,
        "n_in": n_in,
        "iter_max": iter_max,
        "init": init_km,
        "alg": kmeans_alg,
        "aic": km_aic,
        "bic": km_bic,
        "metric": dist_met,
    }

    return {"results": res_df, "cluster_dict": cluster_dict}


def kmedoids(
    assoc_df,
    dist_df,
    res_df,
    nclust=4,
    rand_st=240,
    init_kmed="k-medoids++",
    iter_max=300,
    kmedoids_alg="alternate",
    dist_met="Euclidean",
):
    """Compute the clusters found using the K-means algorithm

    Args:
        assoc_df (pd.Dataframe): Association with the exposure
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe - association data and cluster results
        nclust (int, optional): Number of desired clusters. Defaults to 4.
        rand_st (int, optional): Random number initialisation. Defaults to 240.
        init_kmed (str, optional): Method for initialising the clusters.
                                Default "k-medoids++".
        iter_max (int, optional): Max no. iterations if no cluster convergence.
                                Defaults to 300.
        kmedoids_alg (str, optional): Implementation algorithm.
                                Defaults to "alternate".
        dist_met (str, optional): Distance Metric "Euclidean"(default),
                                or "CosineSimilarity".

    Raises:
        TypeError: If inputs are of the incorrect type
        ValueError: * If init_km is not one of: `k-means++`, `random` or array
                    * If kmeans_alg is not one of: “lloyd”, “elkan”
                    * If dist_met is not "Euclidean" or "CosineSimilarity"

    Returns:
        Dictionary containing:
        * "results"- The results dataframe with appended new clusters
        * "cluster_dict" - The dictionary of the parameters for clustering
    """
    euc_str = "Euclidean"
    cos_str = "CosineSimilarity"
    # TYPE CHECKS
    # Raise TypeError if assoc_df, dist_df, or res_df are not Dataframes
    checks.df_check(assoc_df, "exposure data assoc_df")
    checks.df_check(dist_df, "distance data dist_df")
    checks.df_check(res_df, "results data res_df")
    # Raise TypeError if n_clust, rand_st, or iter_max are not integers
    checks.int_check(nclust, "nclust")
    checks.int_check(rand_st, "rand_st")
    checks.int_check(iter_max, "iter_max")
    # Raise TypeError if init_km is not a string or an array
    if not isinstance(init_kmed, (str, np.ndarray)):
        error_string = """The input init_km should be a string not """ + str(
            type(init_kmed)
        )
        raise TypeError(error_string)
    # Raise TypeError if kmedoids_alg is not a string
    checks.str_check(kmedoids_alg, "kmedoids_alg")
    # VALUE CHECKS
    # Raise ValueError if init_km not one of: `k-medoids++`, `random` or array
    if isinstance(init_kmed, str):
        if init_kmed not in ["k-medoids++", "random"]:
            error_string = """The input init_km should be either `random`,
                `heuristic`,`k-medoids++`, `build` or an array""" + init_kmed
            raise ValueError(error_string)
    # Raise ValueError if kmeans_alg is not one of: “lloyd”, “elkan”
    if kmedoids_alg not in ["alternate", "pam"]:
        error_string = """The input kmeans_alg should be either `alternate`
            or `pam` not """ + kmedoids_alg
        raise ValueError(error_string)
    # Raise ValueError if dist_met not one of "CosineSimilarity" or "Euclidean"
    if dist_met not in [euc_str, cos_str]:
        error_string = """The input dist_met should be either
                        {euc}, {cos} not {dist}""".format(
            euc=euc_str, cos=cos_str, dist=dist_met
        )
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d
        should match the number of rows in the association data %d""" % (
            res_df.shape[0],
            assoc_df.shape[0],
        )
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match
        the number of columns %d in dist_df""" % (
            dist_df.shape[0],
            dist_df.shape[1],
        )
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association
            data %d does not match the number of rows in the distance
            data %d""" % (assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)

    # Run the Clustering
    klab = mtc.method_string("Kmedoids", kmedoids_alg, dist_met, nclust)
    if dist_met == euc_str:
        n_kmedoids = KMedoids(
            n_clusters=nclust,
            metric="euclidean",
            random_state=rand_st,
            init=init_kmed
        ).fit(assoc_df.to_numpy())
        res_df[klab] = n_kmedoids.labels_
        # Centroid distances
        centroids = pd.DataFrame(data=n_kmedoids.cluster_centers_)
        dist_df = dp.centroid_distance(
            cents=centroids, data=assoc_df, membership=res_df[klab]
        )
    elif dist_met == "CosineSimilarity":
        n_kmedoids = KMedoids(
            n_clusters=nclust,
            metric="precomputed",
            init=init_kmed,
            random_state=rand_st,
        ).fit(dist_df.to_numpy())
        res_df[klab] = n_kmedoids.labels_
        # Centroid distances
        centroids = dp.calc_medoids(
            data=dist_df, data_dist=dist_df, membership=res_df[klab]
        )
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=dist_df,
            membership=res_df[klab],
            metric="cosine-sim"
        )
    # Calculate the AIC
    dimension = len(assoc_df.columns)
    kmed_aic = mtc.get_aic(dist_df, dimension)
    kmed_bic = mtc.get_bic(dist_df, dimension=dimension)
    cluster_dict = {
        "alg_name": "K-medoids",
        "nclust": nclust,
        "rand_st": rand_st,
        "iter_max": iter_max,
        "init": init_kmed,
        "alg": kmedoids_alg,
        "aic": kmed_aic,
        "bic": kmed_bic,
        "metric": dist_met,
    }

    return {"results": res_df, "cluster_dict": cluster_dict}


def dbscan(
    assoc_df, dist_df, res_df,
    min_s=5, eps=0.5, db_alg="auto", dist_met="Euclidean"
):
    """Compute the clusters found using the K-means algorithm

    Args:
        assoc_df (pd.Dataframe): Dataframe of association scores
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe - association data and cluster results
        min_s (int, optional): No. of samples (or total weight) in neighborhood
                            for a point to be considered core point. Default 5.
        eps (float, optional): Max distance between two samples for one to be
                            considered as in neighborhood of other. Default 0.5
        db_alg (str, optional): Implementation algorithm. Defaults to "auto".
        dist_met (str, optional): Distance Metric, "Euclidean"(default),
                            or "CosineSimilarity".

    Raises:
        TypeError: assoc_df not a dataframe
        TypeError: dist_df not a dataframe
        TypeError: res_df not a dataframe
        TypeError: min_s not an integer
        TypeError: eps not a float
        TypeError: db_alg not a string
        ValueError: db_alg not one of: `auto`, `ball_tree`, `kd_tree`, `brute`
        ValueError: dist_met is not "Euclidean" or "CosineSimilarity"
        ValueError: the dimensions of the dataframes don't match.

    Returns:
        Dictionary containing:
        * "results"- results dataframe with appended new clusters
        * "cluster_dict" - dictionary of the parameters for this clustering
    """
    euc_str = "Euclidean"
    cos_str = "CosineSimilarity"
    # TYPE CHECKS
    # Raise TypeError if assoc_df, dist_df, or res_df are not Dataframes
    checks.df_check(assoc_df, "exposure data assoc_df")
    checks.df_check(dist_df, "distance data dist_df")
    checks.df_check(res_df, "results data res_df")
    # Raise TypeError if min_s is in an integer
    checks.int_check(min_s, "min_s")
    # Raise TypeError if eps is not a float
    checks.float_check(eps, "eps")
    # Raise TypeError if db_alg is not a string
    checks.str_check(db_alg, "db_alg")
    # VALUE CHECKS
    # ValueError if db_alg not one of: "auto", "ball_tree", "kd_tree", "brute"
    if db_alg not in ["auto", "ball_tree", "kd_tree", "brute"]:
        error_string = """The input db_alg should be either "auto",
            "ball_tree", "kd_tree", "brute" not """ + db_alg
        raise ValueError(error_string)
    # ValueError if dist_met not one of "CosineSimilarity" or "Euclidean"
    if dist_met not in [euc_str, cos_str]:
        error_string = """The input dist_met should be either
                        {euc}, {cos} not {dist}""".format(
            euc=euc_str, cos=cos_str, dist=dist_met
        )
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d
            should match the number of rows in the association
            data %d""" % (res_df.shape[0], assoc_df.shape[0])
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match
            the number of columns %d in dist_df""" % (dist_df.shape[0],
                                                      dist_df.shape[1])
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association
            data %d does not match the number of rows in the distance
            data %d""" % (assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)
    # Run the Clustering
    meth_str = "DBSCAN%d" % (eps * 100)
    if dist_met == euc_str:
        n_dbscan = DBSCAN(
            eps=eps, min_samples=min_s, metric="euclidean", algorithm=db_alg
        ).fit(dist_df.to_numpy())
        nclust = len(np.unique(n_dbscan.labels_))
        klab = mtc.method_string(meth_str, db_alg, dist_met, min_s)
        res_df[klab] = n_dbscan.labels_
        # Centroid distances
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[klab]
        )
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[klab],
            metric="euc"
        )
    elif dist_met == "CosineSimilarity":
        n_dbscan = DBSCAN(
            eps=eps, min_samples=min_s, metric="precomputed", algorithm=db_alg
        ).fit(dist_df.to_numpy())
        nclust = len(np.unique(n_dbscan.labels_))
        klab = mtc.method_string(meth_str, db_alg, dist_met, min_s)
        res_df[klab] = n_dbscan.labels_
        # Centroid distances
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[klab]
        )
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[klab],
            metric="cosine-sim"
        )
    # Calculate the AIC
    dimension = len(assoc_df.columns)
    db_aic = mtc.get_aic(dist_df, dimension=dimension)
    db_bic = mtc.get_bic(dist_df, dimension=dimension)

    cluster_dict = {
        "alg_name": "DBSCAN",
        "nclust": nclust,
        "eps": eps,
        "min_samples": min_s,
        "aic": db_aic,
        "bic": db_bic,
        "metric": dist_met,
    }
    return {"results": res_df, "cluster_dict": cluster_dict}


def gmm(
    assoc_df,
    dist_df,
    res_df,
    n_comps=4,
    rand_st=0,
    max_iter=100,
    cov_type="diag",
    in_pars="random_from_data",
    dist_met="CosineDistance",
):
    """Gaussian Mixture Model clustering

    Args:
        assoc_df (pd.Dataframe): Association with the exposure
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe - association data and cluster results
        n_comps (int, optional): Number of components. Defaults to 4.
        rand_st (int, optional): Random seed. Defaults to 0.
        max_iter (int, optional): Number of EM iterations. Defaults to 100.
        cov_type (str, optional): Covariance type. Defaults to "diag".
        in_pars (str, optional): Method to intialising weights.
                                Default "random_from_data".
        gmm_met (str, optional):  Distance Metric, either "Euclidean",
                                or "CosineDistance"(default)

    Raises:
        TypeError: assoc_df not a dataframe
        TypeError: dist_df not a dataframe
        TypeError: res_df not a dataframe
        TypeError: n_comps not an integer
        TypeError: rand_st not an integer
        TypeError: max_iter not an integer
        TypeError: cov_type not a string
        TypeError: dist_met not a string
        TypeError: in_pars not a string
        ValueError: cov_type not one of: "full", "tied", "diag", "spherical"
        ValueError: in_pars not one of:
                "kmeans", "k-means++", "random", "random_from_data"
        ValueError: dist_met not one of "CosineSimilarity" or "Euclidean"
        ValueError: dataframes are not compatible dimensions

    Returns:
        Dictionary containing:
        * "results"- results dataframe with appended new clusters
        * "cluster_dict" - dictionary of the parameters for this clustering
    """
    euc_str = "Euclidean"
    cos_str = "CosineDistance"
    # TYPE CHECKS
    # Raise TypeError if assoc_df, dist_df, or res_df are not Dataframes
    checks.df_check(assoc_df, "exposure data assoc_df")
    checks.df_check(dist_df, "distance data dist_df")
    checks.df_check(res_df, "results data res_df")
    # Raise TypeError is rand_st, max_iter are not an integers
    checks.int_check(n_comps, "n_comps")
    checks.int_check(rand_st, "rand_st")
    checks.int_check(max_iter, "max_iter")
    # Raise TypeError if cov_type, in_pars, or gmm_met is not a string
    checks.str_check(cov_type, "cov_type")
    checks.str_check(in_pars, "in_pars")
    checks.str_check(dist_met, "gmm_met")
    # VALUE CHECKS
    # ValueError if cov_type not one of: "full", "tied", "diag", "spherical"
    if cov_type not in ["full", "tied", "diag", "spherical"]:
        error_string = """The input cov_type should be either "full", "tied",
            "diag", "spherical" not """ + cov_type
        raise ValueError(error_string)
    # ValueError in_pars not one of:
    # "kmeans", "k-means++", "random", "random_from_data"
    if in_pars not in ["kmeans", "k-means++", "random", "random_from_data"]:
        error_string = """The input in_pars should be either "kmeans",
            "k-means++", "random", "random_from_data" not """ + in_pars
        raise ValueError(error_string)
    # ValueError if gmm_met not one of "CosineSimilarity" or "Euclidean"
    if dist_met not in [euc_str, cos_str]:
        error_string = """The input dist_met should be either
                        {euc}, {cos} not {dist}""".format(
            euc=euc_str, cos=cos_str, dist=dist_met
        )
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d
        should match the number of rows in the association data %d""" % (
            res_df.shape[0],
            assoc_df.shape[0]
        )
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match
            the number of columns %d in dist_df""" % (dist_df.shape[0],
                                                      dist_df.shape[1])
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association
            data %d does not match the number of rows in the distance
            data %d""" % (assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)
    # Run the Clustering
    gmm_lab = mtc.method_string("GMM", cov_type, dist_met, n_comps)
    if dist_met == euc_str:
        gmm = GaussianMixture(
            n_components=n_comps,
            covariance_type=cov_type,
            random_state=rand_st,
            max_iter=max_iter,
            init_params=in_pars
        ).fit(dist_df)
        res_df[gmm_lab] = gmm.predict(dist_df)
        # -----------------------
        # Calculate the centroids
        # Centroid distances
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[gmm_lab]
        )
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[gmm_lab],
            metric="euc"
        )
    elif dist_met == "CosineDistance":
        gmm = GaussianMixture(
            n_components=n_comps,
            covariance_type=cov_type,
            random_state=rand_st,
            max_iter=max_iter,
            init_params=in_pars,
        ).fit(dist_df)
        res_df[gmm_lab] = gmm.predict(dist_df)
        # -----------------------
        # Calculate the centroids
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[gmm_lab]
        )
        # centroids = np.array(n_kmeans_batch.cluster_centers_)
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[gmm_lab],
            metric="cosine-dist",
        )

    # -----------------
    # Calculate the AIC
    dimension = assoc_df.shape[1]
    n_params = n_comps * (2 * dimension - 1)
    gmm_aic = mtc.get_aic(dist_df, dimension=dimension)
    gmm_bic = mtc.get_bic(dist_df, n_params=n_params, dimension=dimension)
    cluster_dict = {
        "alg_name": "Gaussian Mixture Model",
        "nclust": res_df[gmm_lab].nunique(),
        "RandSt": rand_st,
        "NComponents": n_comps,
        "CovarianceType": cov_type,
        "InitParams": in_pars,
        "aic": gmm_aic,
        "bic": gmm_bic,
        "metric": dist_met,
    }
    return {"results": res_df, "cluster_dict": cluster_dict}


def birch(assoc_df, dist_df, res_df,
          thresh=0.5, branch_fac=50, dist_met="Euclidean"):
    """Birch Clustering

    Args:
        assoc_df (pd.Dataframe): Association with the exposure
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe - association data and cluster results
        thresh (float, optional): radius of subcluster obtained by merging a
                            sample and closest subcluster. Default 0.5.
        branch_fac (int, optional): Max no. of CF subclusters in each node.
                            Default 50.
        dist_met (str, optional):  Distance Metric, either "Euclidean",
                            or "CosineDistance"(default)

    Raises:
        TypeError: assoc_df not a dataframe
        TypeError: dist_df not a dataframe
        TypeError: res_df not a dataframe
        TypeError: thresh not a float
        TypeError: branch_fac not an integer
        TypeError: dist_met not a string
        ValueError: dist_met not "Euclidean" or "CosineDistance"
        ValueError: assoc_df does not have the same number of rows as res_df
        ValueError: dist_df doesn't have the same number of rows as columns.
        ValueError: dist_df doesn't have the same number of rows as assoc_df.

    Returns:
        Dictionary containing:
        * "results"- results dataframe with appended new clusters
        * "cluster_dict" - dictionary of the parameters for clustering
    """
    euc_str = "Euclidean"
    cos_str = "CosineDistance"
    # TYPE CHECKS
    # Raise TypeError if assoc_df, dist_df, or res_df are not Dataframes
    checks.df_check(assoc_df, "exposure data assoc_df")
    checks.df_check(dist_df, "distance data dist_df")
    checks.df_check(res_df, "results data res_df")
    # Raise TypeError if thresh is a float
    checks.float_check(thresh, "thresh")
    # Raise TypeError if branch_fac is not an integer
    checks.int_check(branch_fac, "branch_fac")
    # Raise TypeError if dist_met is not a string
    checks.str_check(dist_met, "bir_met")
    # VALUE CHECKS
    # ValueError if dist_met not one of "CosineSimilarity" or "Euclidean"
    if dist_met not in [euc_str, cos_str]:
        error_string = """The input dist_met should be either
                        {euc}, {cos} not {dist}""".format(
            euc=euc_str, cos=cos_str, dist=dist_met
        )
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d
            should match the number of rows in the association
            data %d""" % (res_df.shape[0], assoc_df.shape[0])
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match
            the number of columns %d in dist_df""" % (dist_df.shape[0],
                                                      dist_df.shape[1],
                                                      )
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association data %d
            does not match the number of rows in the distance
            data %d""" % (assoc_df.shape[0], dist_df.shape[0],)
        raise ValueError(error_string)
    # ------------------
    # Run the Clustering
    meth_str = "Birch" + "%d" % (100 * thresh)
    birlab = mtc.method_string(meth_str, "", dist_met, branch_fac)
    if dist_met == euc_str:
        brc = Birch(n_clusters=None,
                    threshold=thresh,
                    branching_factor=branch_fac).fit(dist_df)
        brc_clusts = brc.predict(dist_df)
        res_df[birlab] = brc_clusts
        birch_clust_opts = np.unique(res_df[birlab])
        # ------------------------
        # Calculate the centroids
        # -----------------------
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[birlab]
        )
        # centroids = np.array(n_kmeans_batch.cluster_centers_)
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[birlab],
            metric="euc"
        )
    elif dist_met == cos_str:
        brc = Birch(n_clusters=None,
                    threshold=thresh,
                    branching_factor=branch_fac).fit(dist_df)
        brc_clusts = brc.predict(dist_df)
        res_df[birlab] = brc_clusts
        birch_clust_opts = np.unique(res_df[birlab])
        # ------------------------
        # Calculate the centroids
        # -----------------------
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[birlab]
        )
        # centroids = np.array(n_kmeans_batch.cluster_centers_)
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[birlab],
            metric="cosine-dist",
        )
    # -----------------
    # Calculate the AIC
    dimension = assoc_df.shape[1]
    bir_aic = mtc.get_aic(dist_df, dimension=dimension)
    bir_bic = mtc.get_bic(dist_df, dimension=dimension)
    # -------------------------------
    # Store the clustering parameters
    cluster_dict = {
        "alg_name": "Birch",
        "nclust": len(birch_clust_opts),
        "threshold": thresh,
        "BranchingFactor": branch_fac,
        "aic": bir_aic,
        "bic": bir_bic,
        "metric": dist_met,
    }
    return {"results": res_df, "cluster_dict": cluster_dict}


def kmeans_minibatch(
    assoc_df,
    dist_df,
    res_df,
    nclust=4,
    rand_st=240,
    n_in=50,
    batch_size=30,
    iter_max=300,
    dist_met="Euclidean",
):
    """Kmeans Minibatch clustering

    Args:
        assoc_df (pd.Dataframe): Dataframe for exposure data.
                            rows are the snps, columns are the trait axes.
        dist_df (pd.Dataframe): Dataframe with the distance between all points.
                            rows and columns are snps,
                            values are the distances between the pairs.
        res_df (pd.Dataframe): Dataframe with the clustering results.
                            rows are snps, columns clustering methods
        nclust (int): No. clusters to be assigned. Default 4
        rand_st (int): Random seed. Defaults to 240
        n_in (int): No. random initialisations. Default 50.
        batch_size (int): No. points in each batch. Default 30
        iter_max (int): Max no. iterations. Default 300.
        dist_met (str): Distance Metric. Default "Euclidean".
                        Must be either "Euclidean", or "CosineDistance"

    Raises:
        TypeError: assoc_df not a dataframe
        TypeError: dist_df not a dataframe
        TypeError: res_df not a dataframe
        TypeError: nclust not an integer
        TypeError: batch_size not an integer
        TypeError: iter_max not an integer
        TypeError: dist_met not a string
        ValueError: dist_met not one of "CosineSimilarity" or "Euclidean"
        ValueError: the dataframes do not have compatible dimensions

    Returns:
        Dictionary containing:
        * "results"- results dataframe with appended new clusters
        * "cluster_dict" - dictionary of the parameters for clustering

    """
    euc_str = "Euclidean"
    cos_str = "CosineDistance"
    # TYPE CHECKS
    # TypeError if assoc_df, dist_df, or res_df are not Dataframes
    checks.df_check(assoc_df, "exposure data assoc_df")
    checks.df_check(dist_df, "distance data dist_df")
    checks.df_check(res_df, "results data res_df")
    # TypeError is batch_size, n_in, rand_st and iter_max are not integers
    checks.int_check(batch_size, "batch_size")
    checks.int_check(n_in, "n_in")
    checks.int_check(rand_st, "rand_st")
    checks.int_check(iter_max, "iter_max")
    # TypeError if dist_met is not a string
    checks.str_check(dist_met, "dist_met")
    # VALUE CHECKS
    # ValueError if dist_met not one of "CosineSimilarity" or "Euclidean"
    if dist_met not in [euc_str, cos_str]:
        error_string = """The input bir_met should be either
                        {euc}, {cos} not {dist}""".format(
            euc=euc_str, cos=cos_str, dist=dist_met
        )
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d
        should match the number of rows in the association data %d""" % (
            res_df.shape[0],
            assoc_df.shape[0],
        )
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match
            the number of columns %d in dist_df""" % (dist_df.shape[0],
                                                      dist_df.shape[1]
                                                      )
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association data %d
            does not match the number of rows in the distance
            data %d""" % (assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)
    # ------------------
    # Run the Clustering
    mini_lab = mtc.method_string(
        "MiniBatchKmeans" + "%d" % (batch_size), "", dist_met, nclust
    )
    if dist_met == euc_str:
        mini_clusts = MiniBatchKMeans(
            n_clusters=nclust,
            random_state=rand_st,
            batch_size=batch_size,
            max_iter=iter_max,
            n_init=n_in,
        ).fit(dist_df)
        res_df[mini_lab] = mini_clusts.labels_
        # ------------------------
        # Calculate the centroids
        # -----------------------
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[mini_lab]
        )
        # centroids = np.array(n_kmeans_batch.cluster_centers_)
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[mini_lab],
            metric="euc"
        )
    elif dist_met == cos_str:
        mini_clusts = MiniBatchKMeans(
            n_clusters=nclust,
            random_state=rand_st,
            batch_size=batch_size,
            max_iter=iter_max,
            n_init=n_in,
        ).fit(dist_df)
        res_df[mini_lab] = mini_clusts.labels_
        # ------------------------
        # Calculate the centroids
        # -----------------------
        centroids = dp.calc_medoids(
            data=assoc_df, data_dist=dist_df, membership=res_df[mini_lab]
        )
        # centroids = np.array(n_kmeans_batch.cluster_centers_)
        dist_df = dp.centroid_distance(
            cents=centroids,
            data=assoc_df,
            membership=res_df[mini_lab],
            metric="cosine-dist",
        )
    # -----------------
    # Calculate the AIC
    dimension = assoc_df.shape[1]
    mini_aic = mtc.get_aic(dist_df, dimension=dimension)
    mini_bic = mtc.get_bic(dist_df, dimension=dimension)
    # -------------------------------
    # Store the clustering parameters
    cluster_dict = {
        "alg_name": "K-means mini-batch",
        "nclust": nclust,
        "rand_st": rand_st,
        "iter_max": iter_max,
        "batch_size": batch_size,
        "aic": mini_aic,
        "bic": mini_bic,
        "metric": dist_met,
    }
    return {"results": res_df, "cluster_dict": cluster_dict}
