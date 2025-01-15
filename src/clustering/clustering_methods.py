import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN

from clustering import multi_trait_clustering as mtc
from data_manipulation import data_processing as dp

def kmeans(assoc_df, dist_df, res_df,
                   nclust = 4, rand_st = 240, 
                   n_in = 50, init_km = "k-means++",
                   iter_max = 300, kmeans_alg = "lloyd",
                   dist_met = "Euclidean"):
    """Compute the clusters found using the K-means algorithm

    Args:
        assoc_df (pd.Dataframe): Association with the exposure
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe containing the association data and the cluster results
        nclust (int, optional): Number of desired clusters. Defaults to 4.
        rand_st (int, optional): Random number initialisation. Defaults to 240.
        n_in (int, optional): _description_. Defaults to 50.
        init_km (str, optional): Method for initialising the clusters. Defaults to "k-means++".
        iter_max (int, optional): Maximum number of iterations if no cluster convergence. Defaults to 300.
        kmeans_alg (str, optional): Implementation algorithm. Defaults to "lloyd".
        dist_met (str, optional): Metric used for measuring distance between data-points. Defaults to "Euclidean".

    Raises:
        TypeError: If inputs are of the incorrect type
        ValueError: * If n_in is not an integer or auto
                    * If init_km is not one of: `k-means++`, `random` or an array
                    * If kmeans_alg is not one of: “lloyd”, “elkan”
                    * If dist_met is not "Euclidean" or "CosineSimilarity"
    """
    euc_str = "Euclidean"
    cos_str = "CosineSimilarity"
    # TYPE CHECKS
    # Raise TypeError if exposure data is not entered as a dataframe
    if not isinstance(assoc_df, pd.DataFrame):
        error_string = """The input eff_df should be a dataframe not """ + str(type(assoc_df))
        raise TypeError(error_string)
    # Raise TypeError if distance data is not entered as a dataframe
    if not isinstance(dist_df, pd.DataFrame):
        error_string = """The input dist_df should be a dataframe not """ + str(type(dist_df))
        raise TypeError(error_string)
    # Raise TypeError if results data is not entered as a dataframe
    if not isinstance(res_df, pd.DataFrame):
        error_string = """The input res_df should be a dataframe not """ + str(type(res_df))
        raise TypeError(error_string)
    # Raise TypeError if a non integer number of clusters is entered
    if not isinstance(nclust, int):
        error_string = """The input nclust should be a integer not """ + str(type(nclust))
        raise TypeError(error_string)
    # Raise TypeError if rand_st is not an integer
    if not isinstance(rand_st, int):
        error_string = """The input rand_st should be a integer not """ + str(type(rand_st))
        raise TypeError(error_string)
    # Raise TypeError if n_in is not an integer or string
    if not isinstance(n_in, (str, int)):
        error_string = """The input n_in should be a integer or a string not """ + str(type(n_in))
        raise TypeError(error_string)
    # Raise TypeError if init_km is not a string or an array
    if not isinstance(init_km, (str, np.ndarray)):
        error_string = """The input init_km should be a string not """ + str(type(init_km))
        raise TypeError(error_string)
    # Raise TypeError if iter_max is not an integer
    if not isinstance(iter_max, int):
        error_string = """The input iter_max should be an integer not """ + str(type(iter_max))
        raise TypeError(error_string)
    # Raise TypeError if kmeans_alg is not a string
    if not isinstance(kmeans_alg, str):
        error_string = """The input kmeans_alg should be a string not """ + str(type(kmeans_alg))
        raise TypeError(error_string)
    # VALUE CHECKS
    # Raise ValueError if n_in is not an integer or "auto".
    if isinstance(n_in, str):
        if n_in != "auto":
            error_string = """The input dist_df should be either a integer or auto not """ + n_in
            raise ValueError(error_string)
    # Raise ValueError if init_km is not one of: `k-means++`, `random` or an array
    if isinstance(init_km, str):
        if not init_km in ["k-means++", "random"]:
            error_string = """The input init_km should be either `k-means++`, `random` or an array""" + init_km
            raise ValueError(error_string)
    # Raise ValueError if kmeans_alg is not one of: “lloyd”, “elkan”
    if not kmeans_alg in ["lloyd", "elkan"]:
        error_string = """The input kmeans_alg should be either `lloyd`, `elkan` not """ + kmeans_alg
        raise ValueError(error_string)
    # Raise ValueError if dist_met is not one of "CosineSimilarity" or "Euclidean"
    if not dist_met in [euc_str, cos_str]:
        error_string = """The input dist_met should be either 
                        {euc}, {cos} not {dist}""".format(euc = euc_str, cos = cos_str, dist = dist_met)
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d 
        should match the number of rows in the association data %d"""%(res_df.shape[0], assoc_df.shape[0])
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match 
        the number of columns %d in dist_df"""%(dist_df.shape[0], dist_df.shape[1])
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association 
        data %d does not match the number of rows in the distance data %d"""%(assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)
    
    klab = mtc.method_string('Kmeans', kmeans_alg, dist_met, nclust)
    if dist_met == euc_str:
        n_kmeans = KMeans(n_clusters = nclust,
                  init = init_km,
                  random_state = rand_st,
                  n_init = n_in,
                  max_iter = iter_max,
                  algorithm = kmeans_alg).fit(assoc_df.to_numpy())
        res_df[klab] = n_kmeans.labels_
        # Centroid distances
        centroids = pd.DataFrame(n_kmeans.cluster_centers_, columns = assoc_df.columns)
        dist_df = dp.centroid_distance(cents = centroids, 
                            data = assoc_df,
                            membership = res_df[klab])
    elif dist_met == "CosineSimilarity":
        n_kmeans = KMeans(n_clusters = nclust,
                  init = init_km,
                  random_state = rand_st,
                  n_init = n_in,
                  max_iter = iter_max,
                  algorithm = kmeans_alg).fit(dist_df.to_numpy())
        res_df[klab] = n_kmeans.labels_
        # Centroid distances
        centroids = pd.DataFrame(n_kmeans.cluster_centers_)
        dist_df = dp.centroid_distance(cents = centroids, 
                            data = dist_df,
                            membership = res_df[klab],
                            metric= "cosine-sim")
    # Calculate the AIC
    dimension = len(assoc_df.columns)
    km_aic = mtc.get_aic(dist_df, dimension)
    km_bic = mtc.get_bic(dist_df, dimension=dimension)
    cluster_dict = {
      "nclust": nclust,
      "rand_st": rand_st,
      "n_in": n_in,
      "iter_max": iter_max,
      "init": init_km,
      "alg": kmeans_alg,
      "aic": km_aic,
      "bic": km_bic}

    return({"results": res_df, "cluster_dict": cluster_dict})

def kmedoids(assoc_df, dist_df, res_df,
             nclust = 4, rand_st = 240, init_kmed = "k-medoids++",
             iter_max = 300, kmedoids_alg = "alternate", dist_met = "Euclidean"):
    """Compute the clusters found using the K-means algorithm

    Args:
        assoc_df (pd.Dataframe): Association with the exposure
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe containing the association data and the cluster results
        nclust (int, optional): Number of desired clusters. Defaults to 4.
        rand_st (int, optional): Random number initialisation. Defaults to 240.
        init_kmed (str, optional): Method for initialising the clusters. Defaults to "k-medoids++".
        iter_max (int, optional): Maximum number of iterations if no cluster convergence. Defaults to 300.
        kmedoids_alg (str, optional): Implementation algorithm. Defaults to "alternate".
        dist_met (str, optional): Metric used for measuring distance between data-points. Defaults to "Euclidean".

    Raises:
        TypeError: If inputs are of the incorrect type
        ValueError: * If init_km is not one of: `k-means++`, `random` or an array
                    * If kmeans_alg is not one of: “lloyd”, “elkan”
                    * If dist_met is not "Euclidean" or "CosineSimilarity"
    """
    euc_str = "Euclidean"
    cos_str = "CosineSimilarity"
    # TYPE CHECKS
    # Raise TypeError if exposure data is not entered as a dataframe
    if not isinstance(assoc_df, pd.DataFrame):
        error_string = """The input eff_df should be a dataframe not """ + str(type(assoc_df))
        raise TypeError(error_string)
    # Raise TypeError if distance data is not entered as a dataframe
    if not isinstance(dist_df, pd.DataFrame):
        error_string = """The input dist_df should be a dataframe not """ + str(type(dist_df))
        raise TypeError(error_string)
    # Raise TypeError if results data is not entered as a dataframe
    if not isinstance(res_df, pd.DataFrame):
        error_string = """The input res_df should be a dataframe not """ + str(type(res_df))
        raise TypeError(error_string)
    # Raise TypeError if a non integer number of clusters is entered
    if not isinstance(nclust, int):
        error_string = """The input nclust should be a integer not """ + str(type(nclust))
        raise TypeError(error_string)
    # Raise TypeError if rand_st is not an integer
    if not isinstance(rand_st, int):
        error_string = """The input rand_st should be a integer not """ + str(type(rand_st))
        raise TypeError(error_string)
    # Raise TypeError if init_km is not a string or an array
    if not isinstance(init_kmed, (str, np.ndarray)):
        error_string = """The input init_km should be a string not """ + str(type(init_kmed))
        raise TypeError(error_string)
    # Raise TypeError if iter_max is not an integer
    if not isinstance(iter_max, int):
        error_string = """The input iter_max should be an integer not """ + str(type(iter_max))
        raise TypeError(error_string)
    # Raise TypeError if kmedoids_alg is not a string
    if not isinstance(kmedoids_alg, str):
        error_string = """The input kmeans_alg should be a string not """ + str(type(kmedoids_alg))
        raise TypeError(error_string)
    # VALUE CHECKS
    # Raise ValueError if init_km is not one of: `k-means++`, `random` or an array
    if isinstance(init_kmed, str):
        if not init_kmed in ["k-medoids++", "random"]:
            error_string = """The input init_km should be either `random`, `heuristic`, 
                            `k-medoids++`, `build` or an array""" + init_kmed
            raise ValueError(error_string)
    # Raise ValueError if kmeans_alg is not one of: “lloyd”, “elkan”
    if not kmedoids_alg in ["alternate", "pam"]:
        error_string = """The input kmeans_alg should be either `alternate` or `pam` not """ + kmedoids_alg
        raise ValueError(error_string)
    # Raise ValueError if dist_met is not one of "CosineSimilarity" or "Euclidean"
    if not dist_met in [euc_str, cos_str]:
        error_string = """The input dist_met should be either 
                        {euc}, {cos} not {dist}""".format(euc = euc_str, cos = cos_str, dist = dist_met)
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d 
        should match the number of rows in the association data %d"""%(res_df.shape[0], assoc_df.shape[0])
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match 
        the number of columns %d in dist_df"""%(dist_df.shape[0], dist_df.shape[1])
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association 
        data %d does not match the number of rows in the distance data %d"""%(assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)
    
    # Run the Clustering
    klab = mtc.method_string('Kmedoids', "", dist_met, nclust)
    if dist_met == euc_str:
        n_kmedoids = KMedoids(n_clusters=nclust,
                        metric = "euclidean",
                        random_state = rand_st,
                        init = init_kmed).fit(assoc_df.to_numpy())
        res_df[klab] = n_kmedoids.labels_
        # Centroid distances
        centroids = pd.DataFrame(data = n_kmedoids.cluster_centers_)
        dist_df = dp.centroid_distance(cents = centroids, 
                            data = assoc_df,
                            membership = res_df[klab])
    elif dist_met == "CosineSimilarity":
        n_kmedoids = KMedoids(n_clusters= nclust,
                      metric = "precomputed",
                      init = init_kmed,
                      random_state = rand_st).fit(dist_df.to_numpy())
        res_df[klab] = n_kmedoids.labels_
        # Centroid distances
        centroids = dp.calc_medoids(data = dist_df,
                             data_dist= dist_df,
                             membership = res_df[klab])
        dist_df = dp.centroid_distance(cents = centroids, 
                            data = dist_df,
                            membership = res_df[klab],
                            metric= "cosine-sim")
    # Calculate the AIC
    dimension = len(assoc_df.columns)
    kmed_aic = mtc.get_aic(dist_df, dimension)
    kmed_bic = mtc.get_bic(dist_df, dimension=dimension)
    cluster_dict = {
      "nclust": nclust,
      "rand_st": rand_st,
      "iter_max": iter_max,
      "init": init_kmed,
      "alg": kmedoids_alg,
      "aic": kmed_aic,
      "bic": kmed_bic}

    return({"results": res_df, "cluster_dict": cluster_dict})
def dbscan(assoc_df, dist_df, res_df,
           min_s = 5, eps = 0.5, db_alg = "auto", dist_met = "Euclidean"):
    """Compute the clusters found using the K-means algorithm

    Args:
        assoc_df (pd.Dataframe): Association with the exposure
        dist_df (pd.Dataframe): Distances between all data-points
        res_df (pd.Dataframe): Dataframe containing the association data and the cluster results
        min_s (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. Default to 5.
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default 0.5
        db_alg (str, optional): Implementation algorithm. Defaults to "auto".
        dist_met (str, optional): Metric used for measuring distance between data-points. Defaults to "Euclidean".

    Raises:
        TypeError: If inputs are of the incorrect type
        ValueError: * If db_alg is not one of: `auto`, `ball_tree`, `kd_tree`, `brute`
                    * If dist_met is not "Euclidean" or "CosineSimilarity"
                    * If the dimensions of the dataframes don't match.
    """
    euc_str = "Euclidean"
    cos_str = "CosineSimilarity"
    # TYPE CHECKS
    # Raise TypeError if exposure data is not entered as a dataframe
    if not isinstance(assoc_df, pd.DataFrame):
        error_string = """The input eff_df should be a dataframe not """ + str(type(assoc_df))
        raise TypeError(error_string)
    # Raise TypeError if distance data is not entered as a dataframe
    if not isinstance(dist_df, pd.DataFrame):
        error_string = """The input dist_df should be a dataframe not """ + str(type(dist_df))
        raise TypeError(error_string)
    # Raise TypeError if results data is not entered as a dataframe
    if not isinstance(res_df, pd.DataFrame):
        error_string = """The input res_df should be a dataframe not """ + str(type(res_df))
        raise TypeError(error_string)
    # Raise TypeError if a non integer number of clusters is entered
    if not isinstance(min_s, int):
        error_string = """The input min_s should be a integer not """ + str(type(min_s))
        raise TypeError(error_string)
    # Raise TypeError if rand_st is not an integer
    if not isinstance(eps, float):
        error_string = """The input eps should be a float not """ + str(type(eps))
        raise TypeError(error_string)
    # Raise TypeError if db_alg is not a string
    if not isinstance(db_alg, str):
        error_string = """The input db_alg should be a string not """ + str(type(db_alg))
        raise TypeError(error_string)
    # VALUE CHECKS
    # Raise ValueError if db_alg is not one of: "auto", "ball_tree", "kd_tree", "brute"
    if not db_alg in ["auto", "ball_tree", "kd_tree", "brute"]:
        error_string = """The input kmeans_alg should be either "auto", "ball_tree", "kd_tree", "brute" not """ + db_alg
        raise ValueError(error_string)
    # Raise ValueError if dist_met is not one of "CosineSimilarity" or "Euclidean"
    if not dist_met in [euc_str, cos_str]:
        error_string = """The input dist_met should be either 
                        {euc}, {cos} not {dist}""".format(euc = euc_str, cos = cos_str, dist = dist_met)
        raise ValueError(error_string)
    # DIMENSION CHECKS
    if assoc_df.shape[0] != res_df.shape[0]:
        error_string = """"The number of rows in res_df %d 
        should match the number of rows in the association data %d"""%(res_df.shape[0], assoc_df.shape[0])
        raise ValueError(error_string)
    if dist_df.shape[0] != dist_df.shape[1]:
        error_string = """The number of rows %d should match 
        the number of columns %d in dist_df"""%(dist_df.shape[0], dist_df.shape[1])
        raise ValueError(error_string)
    if dist_df.shape[0] != assoc_df.shape[0]:
        error_string = """The number of rows in the association 
        data %d does not match the number of rows in the distance data %d"""%(assoc_df.shape[0], dist_df.shape[0])
        raise ValueError(error_string)
    # Run the Clustering
    if dist_met == euc_str:
        n_dbscan = DBSCAN(eps = eps,
                      min_samples = min_s,
                      metric = "euclidean",
                      algorithm= db_alg).fit(dist_df.to_numpy())
    elif dist_met == "CosineSimilarity":
        n_dbscan = DBSCAN(eps = eps,
                      min_samples = min_s,
                      metric = "precomputed",
                      algorithm= db_alg).fit(dist_df.to_numpy())
    nclust = len(np.unique(n_dbscan.labels_))
    klab = mtc.method_string("DBSCAN%d"%(eps * 100), db_alg, dist_met, min_s)
    res_df[klab] = n_dbscan.labels_
    # Centroid distances
    centroids = dp.calc_medoids(data = assoc_df,
                             data_dist= dist_df,
                             membership = res_df[klab])
    dist_df = dp.centroid_distance(cents = centroids, 
                            data = assoc_df,
                            membership = res_df[klab])
    # Calculate the AIC
    dimension = len(assoc_df.columns)
    db_aic = mtc.get_aic(dist_df, dimension)
    db_bic = mtc.get_bic(dist_df, dimension=dimension)

    cluster_dict = {
        "nclust": nclust,
        "eps": eps,
        "min_samples": min_s,
        "aic": db_aic,
        "bic": db_bic,
        "metric": dist_met
    }
    return({"results": res_df, "cluster_dict": cluster_dict})

