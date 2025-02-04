from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from itertools import product

from multitraitclustering import data_setup as ds


def compare_results_list_to_external(clust_df, external_df, external_lab):
    """Compare clustering results to an external method.

    Take all clustering results in `clust_df` and compare to
    clusters in `external_df` given by `external_lab`. Computes
    the comparison as the percentage of the number of
    points in the intersection over the union of cluster pairs.
    Return a dictionary of Dataframes corresponding to each clustering method.

    Args:
        clust_df (pd.DataFrame): Main dataframe containing all the
                                 computed clustering results.
        external_df (pd.DataFrame): External dataframe to compare the
                                    cluster membership to.
        external_lab (str, int, float): Label for the column in external_df
                                    to use for comparison results.

    Raise:
        TypeError: clust_df is not a dataframe
        TypeError: eternal_df is not a dataframe
        TypeError: external_lab is not a string
        ValueError: external_lab is not an available column in external_df
        ValueError: no overlapping indices between clust_df and external_df

    Returns: Dictionary with terms
        * comp_dfs, dictionary of Dataframes with comparison percentage grids
        * cluster_matchings, dictionary of best match cluster pairs Dataframes
        * overlap_score, dictionary of the overlap scores

    """
    # Validate the inputs
    # TypeErrors
    if not isinstance(clust_df, pd.DataFrame):
        error_string = "Input clust_df should be a Dataframe not "\
            + str(type(clust_df))
        raise TypeError(error_string)
    if not isinstance(external_df, pd.DataFrame):
        error_string = "Input external_df should be a Dataframe not "\
              + str(type(external_df))
        raise TypeError(error_string)
    if not isinstance(external_lab, (str, int, float)):
        error_string = "Input external should be a string not "\
            + str(type(external_lab))
        raise TypeError(error_string)
    # ValueErrors
    if external_lab not in external_df.columns:
        error_string = """Input external_lab {lab} should be a column in
        external_df. Options: {col_list}""".format(
            lab=external_lab, col_list=str(external_df.columns)
        )
        raise ValueError(error_string)
    snps = clust_df.index
    external_snps = external_df.index
    if len(snps.intersection(external_snps)) == 0:
        error_string = """There are no overlapping indices between the
        two clustering Dataframes."""
        raise ValueError(error_string)
    compare_per_external = {}
    cluster_matchings = {}
    score_to_external = {}
    external_crop_df = external_df.loc[snps, :]
    for meth_key in clust_df.columns:
        compare_arr = ds.compare_df1_to_df2(
            clust_df, external_crop_df, meth_key, external_lab
        )
        comp_df = pd.DataFrame(
            data=compare_arr,
            index=[meth_key + "_" + str(i)
                   for i in clust_df.loc[:, meth_key].unique()],
            columns=[
                external_lab + "_" + str(j)
                for j in external_crop_df.loc[:, external_lab].unique()
            ],
        )
        compare_per_external[meth_key] = calc_per_from_comp(comp_df)
        cluster_matchings[meth_key] = overlap_pairs(
            compare_per_external[meth_key], meth_key
        )
        score_to_external[meth_key] = overlap_score(
            compare_per_external[meth_key]
        )

    out_dict = {
        "comp_dfs": compare_per_external,
        "cluster_matchings": cluster_matchings,
        "overlap_scores": score_to_external,
    }
    return out_dict


def centroid_distance(cents, data, membership, metric="euc"):
    """Calculate the distance between points in a cluster and the centroid

    Args:
        cents (pd.Dataframe): rows correspond to the cluster labels,
                            the columns are the axes labels for the data-space
                            and the values form the position of the centre
        data (pd.Dataframe): rows are snps giving the individual data-points,
                            the columns are the axes labels for the data-space
                            and the values form the positions of the data-point
        membership (pd.Series): indexes are snps, values are the cluster labels
        metric (str, optional): String to indicate which metric to use.
                            Defaults to "euc" for the Euclidean distance.

    Returns:
        distance_df (pd.Dataframe): The distance between each data-point and
                            cluster centre for the cluster it is assigned to.
    """
    met_opts = ["euc", "cosine-sim", "cosine-dist"]
    # Check the input types
    if not isinstance(cents, pd.DataFrame):
        error_string = "The cents input should be a dataframe not "\
            + str(type(cents))
        raise TypeError(error_string)
    if not isinstance(data, pd.DataFrame):
        error_string = "The data input should be a dataframe not "\
            + str(type(data))
        raise TypeError(error_string)
    if not isinstance(membership, pd.Series):
        error_string = "The membership input should be a series not "\
            + str(type(membership))
        raise TypeError(error_string)
    if not isinstance(metric, str):
        error_string = "The metric input should be a string not "\
            + str(type(metric))
        raise TypeError(error_string)
    # Check the metric has a valid value
    if metric not in met_opts:
        error_string = (
            "metric should be one of " + str(met_opts) + " not " + str(metric)
        )
        raise ValueError(error_string)
    # Check the dimensions match
    if membership.shape[0] != data.shape[0]:
        error_string = """Dimension mismatch. Membership has {mem_points}
                        points, and needs {data_points},
                        the number data points in data.""".format(
            mem_points=str(membership.shape[0]), data_points=str(data.shape[0])
        )
        raise ValueError(error_string)
    if data.shape[1] != cents.shape[1]:
        error_string = """Dimension mismatch. The cents have {cent_axes} axes,
                        and needs {trait_axes} columns to match the data.
                        """.format(
            cent_axes=str(cents.shape[1]), trait_axes=str(data.shape[1])
        )
        raise ValueError(error_string)
    # Check there is a centre for all clusters in membership.
    if membership.nunique() > cents.shape[0]:
        error_string = """There are centres defined for {nclust} clusters
                        but there are {clust_labs} cluster labels in the
                        membership.""".format(
            nclust=str(cents.shape[0]), clust_labs=str(membership.nunique())
        )
        raise ValueError(error_string)
    if not all(m in cents.index for m in membership.unique()):
        error_string = """There are clusters in the membership list {mems},
                        which do not have a corresponding centre in {cent_list}
                        """.format(
            mems=str(membership.unique()), cent_list=str(cents.index)
        )
        raise ValueError(error_string)
    distance_df = pd.DataFrame(
        index=membership.index, columns=["clust_dist", "clust_num"]
    )
    i = 0
    if metric == "euc":
        # Euclidean centres
        for snp, row in data.iterrows():
            clust_num = membership[snp]
            clust_cent = cents.loc[clust_num, :]
            distance_df.loc[snp, "clust_dist"] = np.linalg.norm(
                clust_cent - row
            )
            distance_df.loc[snp, "clust_num"] = clust_num
            i += 1
    elif metric == "cosine-dist":
        # CosineDistance centres
        for snp, row in data.iterrows():
            clust_num = membership[snp]
            clust_cent = cents.loc[clust_num, :]
            distance_df.loc[snp, "clust_dist"] = cosine(clust_cent, row)
            distance_df.loc[snp, "clust_num"] = clust_num
            i += 1
    elif metric == "cosine-sim":
        for snp, row in data.iterrows():
            clust_num = membership[snp]
            clust_cent = cents.loc[clust_num, :]
            distance_df.loc[snp, "clust_dist"] = cosine(clust_cent, row) - 1
            distance_df.loc[snp, "clust_num"] = clust_num
            i += 1
    return distance_df


def calc_medoids(data, data_dist, membership):
    """Calculate the co-ordinates of the medoids for each cluster.

    The medoid is given by the point with the minimal distance to the
    other points in the cluster. The is calculated by finding the total
    distance to the other cluster points for each point, then returning
    the point whose total is the minimum. This varies from the centroids
    as it always returns a point which is in the cluster.

    Args:
        data (pd.Dataframe): The original data used to create the clusters.
                            rows are the points, columns are the axes.
        data_dist (pd.Dataframe): The distance between all pairs of SNPs.
                            rows and columns are the SNPs and cell-values
                            are the distance.
        membership (pd.Series): SNPs correspond to the rows,
                            the column in the cluster results, the cell
                            values are the value corresponding to the cluster.

    Returns:
        medoids_df (pd.Dataframe): Each row corresponds to the medoid for the
                                corresponding cluster. The columns correspond
                                to the data axes.
    """
    # Check data is a dataframe
    if not isinstance(data, pd.DataFrame):
        error_string = "The input data should be a dataframe not "\
            + str(type(data))
        raise TypeError(error_string)
    # Check data_dist is a dataframe
    if not isinstance(data_dist, pd.DataFrame):
        error_string = "The input data_dist should be a dataframe not "\
            + str(type(data_dist))
        raise TypeError(error_string)
    # Check data is a dataframe
    if not isinstance(membership, pd.Series):
        error_string = "The input membership should be a dataframe not "\
            + str(type(membership))
        raise TypeError(error_string)
    # Check the dimensions match for data and dist
    if data.shape[0] != data_dist.shape[0]:
        error_string = """The data dataframe has %d data points which
                        does not match the %d data points in dist_df""" % (
            data.shape[0],
            data_dist.shape[0],
        )
        raise ValueError(error_string)
    # Check the dimensions match for dist rows and columns
    if data_dist.shape[0] != data_dist.shape[1]:
        error_string = """The dist dataframe has %d rows which does not
                       match the %d columns""" % (
            data_dist.shape[0],
            data_dist.shape[1],
        )
        raise ValueError(error_string)
    if data.shape[0] != len(membership):
        error_string = """The dist dataframe has %d rows which
                        does not match the %d data points in membership""" % (
            data_dist.shape[0],
            len(membership),
        )
        raise ValueError(error_string)

    medoids_out = {}
    for c_num in membership.unique():
        members = membership[membership == c_num].index
        dist_crop = data_dist.loc[members, :]
        dist_crop = dist_crop.loc[:, members]
        medoid = np.argmin(dist_crop.sum(axis=0))
        medoids_out[c_num] = data.iloc[medoid]
    medoids_df = pd.DataFrame.from_dict(medoids_out).transpose()

    return medoids_df


def overlap_score(comp_percent_df):
    """Compute overlap score from percentage overlaps between cluster pairings.

    Find the best matching between the cluster methods by finding the largest
    percentage overlap for each column. This gives the clusters in method 1
    which best match the clusters in method 2.

    Args:
        comp_percent_df (pd.Dataframe): Rows are clusters for the first method.
                                    Columns are clusters for the second method.
                                    Each cell value is the number of points in
                                    the intersection of the cluster pairs
                                    divided by number of points in the union

    Returns:
        overlap_score (float): The mean of the overlap for the best matches.
    """
    # Check data is a dataframe
    if not isinstance(comp_percent_df, pd.DataFrame):
        error_string = "The input comparison data should be a dataframe not "\
            + str(type(comp_percent_df))
        raise TypeError(error_string)
    overlaps = np.amax(comp_percent_df, axis=1)
    overlap_score = overlaps.mean()
    return overlap_score


def overlap_pairs(comp_percent_df, meth_lab, meth_sec_lab="paper"):
    """Find cluster labels for best match pairs between clustering methods

    Args:
        comp_percent_df (pd.Dataframe): Rows are clusters for first method.
                                    Columns are clusters for second method.
                                    cell value is no. of points in intersection
                                    of the cluster pairs divided by the
                                    no. of points in the union.
        meth_lab (string): Label clustering method
        meth_sec_lab (string): Label second cluster method, default = "paper"

    Returns:
        clust_match_df (pd.Dataframe): Col 1 clusters for first cluster method
                                       Col 2 clusters for second cluster method
                                       which best match the clusters from first
                                       Col 3 no. of points in intersection
                                       between the clustering methods divided
                                       by the no. in the union.
    """
    # Check input comp_percent_df is a Dataframe
    if not isinstance(comp_percent_df, pd.DataFrame):
        error_string = "The input comparison data should be a dataframe not "\
            + str(type(comp_percent_df))
        raise TypeError(error_string)
    # Check meth_lab is a string
    if not isinstance(meth_lab, str):
        error_string = "The input method label should be a string "\
            + str(type(meth_lab))
        raise TypeError(error_string)
    # Check meth_sec_lab is a string
    if not isinstance(meth_lab, str):
        error_string = "The input second method label should be a string "\
            + str(type(meth_sec_lab))
        raise TypeError(error_string)
    pairs = np.argmax(comp_percent_df, axis=1)
    overlaps = np.amax(comp_percent_df, axis=1)
    clust_match_df = pd.DataFrame(
        data={
            "cluster_" + meth_lab: np.arange(comp_percent_df.shape[0]),
            "cluster_" + meth_sec_lab: pairs,
            "overlap": overlaps,
        }
    )
    return clust_match_df


def calc_per_from_comp(comp_vals):
    """Calculate the percentage overlap between clusters
    from the number of points in

    Args:
        comp_vals (pd.Dataframe): rows - clusters for first cluster method
                                  columns - clusters for second cluster method,
                                  cell values are no. of points in both

    Percentage is calculated by taking the number of points in the
    intersection of the clustering methods (the cell values) and dividing by
    the number of points in the union (the sum of the number of points in the
    full column and full row).

    Returns:
        comp_out (pd.Dataframe): rows - clusters for first cluster method
                                 columns - clusters for second cluster method,
                                 cell values are percentage of points in both
    """
    if not isinstance(comp_vals, pd.DataFrame):
        error_string = """comparison values input should be in the form
            of a dataframe not """ + str(type(comp_vals))
        raise TypeError(error_string)
    nrows, ncols = comp_vals.shape
    comp_out_dat = np.zeros((nrows, ncols))
    for i, j in product(range(nrows), range(ncols)):
        union = (
            comp_vals.iloc[i, :].sum()
            + comp_vals.iloc[:, j].sum()
            - comp_vals.iloc[i, j]
        )
        comp_out_dat[i, j] = comp_vals.iloc[i, j] / union
    comp_out = pd.DataFrame(
        index=comp_vals.index, columns=comp_vals.columns, data=comp_out_dat
    )
    return comp_out
