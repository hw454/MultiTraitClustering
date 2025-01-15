from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from itertools import product

def centroid_distance(cents, data, membership,
                      metric = "euc"):
    """Calculate the distance between points in a cluster and the centroid assigned
    to the cluster using the defined metric for distance.

    Args:
        cents (pd.Dataframe): rows correspond to the cluster labels, 
                                the columns are the axes labels for the data-space
                                and the values form the position of the centre of the cluster in the data-space.
        data (pd.Dataframe): rows are snps giving the individual data-points,
                                the columns are the axes labels for the data-space
                                and the values form the positions of the data-point in the data-space.
        membership (pd.Series): indexes are snps and the values are the cluster labels.
        metric (str, optional): String to indicate which metric to use for distance. Defaults to "euc" for the Euclidean distance.

    Returns:
        distance_df (pd.Dataframe): The distance between each data-point and the cluster centre for the cluster it is assigned to.
    """
    met_opts = ["euc", "cosine-sim", "cosine-dist"]
    # Check the input types
    if not isinstance(cents, pd.DataFrame):
        error_string = "The cents input should be a dataframe not " + str(type(cents))
        raise TypeError(error_string)
    if not isinstance(data, pd.DataFrame):
        error_string = "The data input should be a dataframe not " + str(type(data))
        raise TypeError(error_string)
    if not isinstance(membership, pd.Series):
        error_string = "The membership input should be a series not " + str(type(membership))
        raise TypeError(error_string)
    if not isinstance(metric, str):
        error_string = "The metric input should be a string not " + str(type(metric))
        raise TypeError(error_string)
    # Check the metric has a valid value
    if not metric in met_opts:
        error_string = "The metric should be one of " + str(met_opts) + "  not " + str(metric)
        raise ValueError(error_string)
    # Check the dimensions match
    if membership.shape[0] != data.shape[0]:
        error_string = """Dimension mismatch. The membership series has {mempoints} points, 
                        and needs {datapoints}, 
                        the number data points in data.""".format(mempoints = str(membership.shape[0]),
                                                                  datapoints = str( data.shape[0]) )
        raise ValueError(error_string)
    if data.shape[1] != cents.shape[1]:
        error_string = """Dimension mismatch. The cents have {centaxes} axes, and needs 
                        {traitaxes} columns to match the data.""".format(centaxes = str(cents.shape[1]), 
                                                                 traitaxes = str(data.shape[1]))
        raise ValueError(error_string)
    # Check there is a centre for all clusters in membership.
    if membership.nunique() > cents.shape[0]:
        error_string = """There are centres defined for {nclust} clusters but there are {clustlabs} 
        cluster labels in the membership.""".format(nclust = str(cents.shape[0]), 
                                                    clustlabs = str(membership.nunique()))
        raise ValueError(error_string)
    distance_df = pd.DataFrame(index = membership.index, columns = ["clust_dist", "clust_num"])
    i = 0
    if metric == "euc":
        for snp, row in data.iterrows():
            #print(snp, membership.index, data.index, data.shape[0] != membership.shape[0])
            clust_num = membership[snp]
            clust_cent = cents.iloc[clust_num, :]
            distance_df.loc[snp, "clust_dist"] = np.linalg.norm(clust_cent-row)
            distance_df.loc[snp, "clust_num"] = clust_num
            i += 1
    elif metric == "cosine-dist":
        for snp, row in data.iterrows():
            clust_num = membership[snp]
            clust_cent = cents.iloc[clust_num]
            distance_df.loc[snp, "clust_dist"] = cosine(clust_cent,row)
            distance_df.loc[snp, "clust_num"] = clust_num
            i += 1
    elif metric == "cosine-sim":
        for snp, row in data.iterrows():
            clust_num = membership[snp]
            clust_cent = cents.iloc[clust_num]
            distance_df.loc[snp, "clust_dist"] = cosine(clust_cent,row)-1
            distance_df.loc[snp, "clust_num"] = clust_num
            i += 1
    return distance_df


def calc_medoids(data, data_dist, membership):
    """ Calculate the co-ordinates of the medoids for each cluster.

    The medoid is given by the point with the minimal distance to the other points in the cluster.
    The is calculated by finding the total distance to the other cluster points for each point, then 
    returning the point whose total is the minimum. This varies from the centroids as it always returns
    a point which is in the cluster.

    Args:
        data (pd.Dataframe): _description_
        data_dist (pd.Dataframe): The distance between all pairs of SNPs. 
                                    rows and columns are the SNPs and cell-values are the distance.
        membership (pd.Series): SNPs correspond to the rows, 
                                    the column in the cluster results, 
                                    the cell values are the value corresponding to the cluster.

    Returns:
        medoids_df (pd.Dataframe): Each row corresponds to the medoid for the corresponding cluster.
                                   The columns correspond to the data axes.
    """
    # Check data is a dataframe
    if not isinstance(data, pd.DataFrame):
        error_string = "The input data should be a dataframe not " + str(type(data))
        raise TypeError(error_string)
    # Check data_dist is a dataframe
    if not isinstance(data_dist, pd.DataFrame):
        error_string = "The input data_dist should be a dataframe not " + str(type(data_dist))
        raise TypeError(error_string)
    # Check data is a dataframe
    if not isinstance(membership, pd.Series):
        error_string = "The input membership should be a dataframe not " + str(type(membership))
        raise TypeError(error_string)
    # Check the number of rows in membership matches the number in the data
    if not membership.shape[0]==data.shape[0]:
        error_string = """The number of data points in the membership {memrows}
                        does not match the number of data points in the data 
                        {datarows}.""".format(memrows = str(membership.shape[0]), 
                                              datarows = str(data.shape[0]))
        raise ValueError(error_string)

    medoids_out = {}
    snp_list = { snp : i for i, snp in enumerate(membership.index)}
    for c_num in membership.unique():
        members = membership[membership == c_num].index
        mem_nums = [snp_list[mem] for mem in members]
        dist_crop = data_dist.iloc[mem_nums,:]
        dist_crop = dist_crop.iloc[:, mem_nums]
        medoid = np.argmin(dist_crop.sum(axis=0))
        medoids_out[c_num] = data.iloc[medoid]
    medoids_df = pd.DataFrame.from_dict(medoids_out).transpose()
    
    return(medoids_df)

def overlap_score(comp_percent_df):
    """ Compute the overlap score from the percentage overlaps between cluster pairings.

    Find the best matching between the cluster methods by finding the largest percentage overlap for each column. This
    gives the clusters in method 1 which best match the clusters in method 2.

    Args:
        comp_percent_df (pd.Dataframe): Rows are the clusters for the first method. Columns are the clusters for the second method.
                                        Each cell value is the number of points in the intersection of the cluster pairs divided by the number of points in the union.

    Returns:
        overlap_score (float): The mean of the overlap for the best matches.
    """
    # Check data is a dataframe
    if not isinstance(comp_percent_df, pd.DataFrame):
        error_string = "The input comparison data should be a dataframe not " + str(type(comp_percent_df))
        raise TypeError(error_string)
    overlaps = np.amax(comp_percent_df, axis = 1)
    overlap_score = overlaps.mean()
    return overlap_score

def overlap_pairs(comp_percent_df, meth_lab, meth_sec_lab = "paper"):
    """Find the cluster labels for the best matched cluster pairs between two different clustering methods. 

    Args:
        comp_percent_df (pd.Dataframe): Rows are the clusters for the first method. Columns are the clusters for the second method.
                                        Each cell value is the number of points in the intersection of the cluster pairs divided by the number of points in the union.
        meth_lab (string): Label for the clustering method
        meth_sec_lab (string): Label for the second cluster method, default = "paper".

    Returns:
        clust_match_df (pd.Dataframe): Column 1 the clusters for the first clustering method. 
                                       Column 2 the clusters from the second clustering method which best match the clusters from the first.
                                       Column 3 the number of points in the intersection between the clustering methods divided by the number in the union.
    """
    # Check input comp_percent_df is a Dataframe
    if not isinstance(comp_percent_df, pd.DataFrame):
        error_string = "The input comparison data should be a dataframe not " + str(type(comp_percent_df))
        raise TypeError(error_string)
    # Check meth_lab is a string
    if not isinstance(meth_lab, str):
        error_string = "The input method label should be a string " + str(type(meth_lab))
        raise TypeError(error_string)
    # Check meth_sec_lab is a string
    if not isinstance(meth_lab, str):
        error_string = "The input second method label should be a string " + str(type(meth_sec_lab))
        raise TypeError(error_string)
    pairs = np.argmax(comp_percent_df, axis = 1)
    overlaps = np.amax(comp_percent_df, axis = 1)
    clust_match_df = pd.DataFrame(
        data = {
        "cluster_" + meth_lab: np.arange(comp_percent_df.shape[0]),
        "cluster_" + meth_sec_lab: pairs,
        "overlap": overlaps})
    return clust_match_df

def calc_per_from_comp(comp_vals):
    """Calculate the percentage overlap between clusters
    from the number of points in

    Args:
        comp_vals (pd.Dataframe): rows correspond to the clusters for one clustering method
                                  columns correspond to the clusters for a second clustering method,
                                  cell values are the number of points present in both clustering methods.

    Percentage is calculated by taking the number of points in the intersection of the clustering methods 
    (the cell values) and dividing by the number of points in the union (the sum of the number of points 
    in the full column and full row).        

    For cell (i,j):
        comp_out[i,j] = comp_vals[i,:].sum() + comp_vals[:,j].sum() - comp_vals[i,j]      
    
    Returns:
        comp_out (pd.Dataframe): rows correspond to the clusters for one clustering method
                                  columns correspond to the clusters for a second clustering method,
                                  cell values are the percentage of points present in both clustering methods.
    """
    if not isinstance(comp_vals, pd.DataFrame):
        error_string = "The comparison values input should be in the form of a dataframe not " + str(type(comp_vals))
        raise TypeError(error_string)    
    nrows, ncols = comp_vals.shape
    comp_out_dat = np.zeros((nrows,ncols))
    for i, j in product(range(nrows), range(ncols)):
        union = comp_vals.iloc[i,:].sum() + comp_vals.iloc[:,j].sum() - comp_vals.iloc[i,j]
        comp_out_dat[i,j] = comp_vals.iloc[i,j] / union
    comp_out = pd.DataFrame(index = comp_vals.index, 
                            columns = comp_vals.columns,
                            data = comp_out_dat)
    return comp_out
def calc_medoids(data, data_dist, membership):
    # Check input data is a Dataframe
    if not isinstance(data, pd.DataFrame):
        error_string = "The input data should be a dataframe not " + str(type(data))
        raise TypeError(error_string)
    # Check data_dist is a Dataframe
    if not isinstance(data_dist, pd.DataFrame):
        error_string = "The input data_dist should be a dataframe not " + str(type(data_dist))
        raise TypeError(error_string)
    # Check input comp_percent_df is a Dataframe
    if not isinstance(membership, pd.Series):
        error_string = "The input membership should be a series not " + str(type(membership))
        raise TypeError(error_string)
    # Check the dimensions match for data and dist
    if data.shape[0] != data_dist.shape[0]:
        error_string = """The data dataframe has %d 
                        data points which does not match the %d data points in 
                        dist_df"""%(data.shape[0], data_dist.shape[0])
        raise ValueError(error_string)
    # Check the dimensions match for dist rows and columns
    if data_dist.shape[0] != data_dist.shape[1]:
        error_string = """The dist dataframe has %d 
                        rows which does not match the %d 
                        columns"""%(data_dist.shape[0], data_dist.shape[1])
        raise ValueError(error_string)
    if data.shape[0] != len(membership):
        error_string = """The dist dataframe has %d 
                        rows which does not match the %d 
                        datapoints in membership"""%(data_dist.shape[0], len(membership))
        raise ValueError(error_string)
    medoids_out = {}
    for c_num in membership.unique():
        members = membership[membership == c_num].index
        dist_crop = data_dist.loc[members,:]
        dist_crop = dist_crop.loc[:, members]
        medoid = np.argmin(dist_crop.sum(axis=0))
        medoids_out[c_num] = data.iloc[medoid]
    medoids_df = pd.DataFrame.from_dict(medoids_out).transpose()
    return(medoids_df)
