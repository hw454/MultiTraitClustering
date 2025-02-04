from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

def compare_df1_to_df2(clust_df1, clust_df2, lab1, lab2):
    """The function `compare_df1_to_df2` compares the cluster assignments between two clustering methods.
    It creates a numpy array `res_arr` which represents the number of points in the intersection between 
    clusters. 

    In `res_arr` the rows correspond to the clusters in `clust_df1` and the 
    columns the clusters in `clust_df2`. The value in each cell 
    of the array is the number of points in cluster `i` for
    the first method and cluster `j` for the second method.

    Args:
        clust_df1 (pd.Dataframe): The column `lab1` of `clust_df1` gives the cluster label assigned to each snp. 
        clust_df2 (pd.Dataframe):  The column `lab2` of `clust_df2` gives the cluster label assigned to each snp.
        lab1 (string): The name of the column of `clust_df1` which gives the clusters assigned to each snp.
        lab2 (string): The name of the column of `clust_df1` which gives the clusters assigned to each snp.

    Returns:
        res_arr (numpy array): with values corresponding to the number of points in the intersection
        of the clusters between clustering methods.
    """

    # Check that both cluster dataframes are Dataframes
    if not isinstance(clust_df1, pd.DataFrame):
        error_string = "The cluster data should be a pandas dataframe not " + str(type(clust_df1))
        raise TypeError(error_string)
    if not isinstance(clust_df2, pd.DataFrame):
        error_string = "The cluster data should be a pandas dataframe not " + str(type(clust_df2))
        raise TypeError(error_string)
    # Check the column label is a string
    if not isinstance(lab1, str):
        error_string = "The column label should be a string " + str(type(lab1))
        raise TypeError(error_string)
    if not isinstance(lab2, str):
        error_string = "The column label should be a string " + str(type(lab2))
        raise TypeError(error_string)
    # Check the column label is a valid value
    if not lab1 in clust_df1.columns:
        error_string = "The column label " + lab1 + " does not match any of the columns " + clust_df1.columns
        raise ValueError(error_string)
    if not lab2 in clust_df2.columns:
        error_string = "The column label " + lab2 + " does not match any of the columns " + clust_df2.columns
        raise ValueError(error_string)
    
    # Since the clusters may have none integer labels create a mapping between the terms and their position. 
    cnum1 = clust_df1[lab1].unique()
    cnum2 = clust_df2[lab2].unique()
    # Create a mapping between the cluster names and the position in the array.
    # Since some clusters are named "Null" or "Junk" the number can not always be used
    # The labels for the clusters are converted to strings to account for NaN and None clusters.
    cnum1_pos = {str(cn): idx for idx, cn in enumerate(cnum1)}
    cnum2_pos = {str(cn): idx for idx, cn in enumerate(cnum2)}
    # Initialise the results array
    res_arr = np.zeros((len(cnum1), len(cnum2)))
    snp_list = clust_df1.index
    for snp in snp_list:
        # Get the cluster number for the snp for each clustering method
        cn1 = clust_df1[lab1].loc[snp]
        cn2 = clust_df2[lab2].loc[snp]
        # Get the position in the array for the cluster
        i = cnum1_pos[str(cn1)]
        j = cnum2_pos[str(cn2)]
        # Increase the count for the intersection of cn1 and cn2 each time a snp is present
        res_arr[i,j] += 1
    return res_arr


def calc_sum_sq (c_num, data):
    """Calculate the sum of the square of the cluster distances for the cluster `clust_num`.

    .. math::

        c_n = \text{ cluster center }c_n\text{ for cluster }n.
        s = \sigma_{\forall p_i \in\text{ cluster}n} ||p_i - c_n||^2, 
        
    Args:
        c_num (string): cluster label
        data (pd.Dataframe): Rows correspond to snps and contains the columns:
                                * `clust_num` indicating the cluster label for each snp,
                                * `clust_dist` the distance between each snp and the cluster centre.
    
    Returns:
        s (float): the sum of the square of the distances between the snps and the cluster centre for 
        cluster number `clust_num`.

    """
    # Check the inputs
    if not isinstance(c_num, int):
        error_string = "The number of clusters should be an integer not " + str(type(c_num))
        raise TypeError(error_string)
    if not isinstance(data, pd.DataFrame):
        error_string = "The data for calculating should be a Dataframe not " + str(type(data))
        raise TypeError(error_string)
    if not "clust_dist" in data.columns:
        error_string = "The data should have a column named `clust_dist`."
        raise ValueError(error_string)
    if not "clust_num" in data.columns:
        error_string = "The data should have a column named `clust_num`."
        raise ValueError(error_string)
    if not c_num in data["clust_num"].unique():
        nums = [ int(cn) for cn in data["clust_num"].unique() if str(cn).isnumeric()]
        # If c_num is a number is should be in the range of the minimum and maximum cluster numbers
        if not min(nums) <= c_num <= max(nums):
            error_string = "The requested cluster distances correspond to a cluster number not in the available range."
            raise IndexError(error_string)

    
    sq = data[data["clust_num"] == c_num]["clust_dist"]**2
    s = sum(sq)
    return(s)

def mat_dist(df, met = "euclidean"):
    """ Creates a matrix corresponding to the distances between all points in the dataframe.

    Args:
        df (pd.Dataframe): Rows correspond to data-points
        met (str, optional): Label for distance metric. Defaults to "euclidean".

    Returns:
        dist_matrix (np.Array): Array of distances between data-points.
    """

    met_opts = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
                   'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1',
                   'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                   'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    if not isinstance(df, pd.DataFrame):
        error_string = "The data for calculating should be a Dataframe not " + str(type(df))
        raise TypeError(error_string)
    if not isinstance(met, str):
        error_string = "The distance metric label should be a string " + str(type(met))
        raise TypeError(error_string)
    met = met.lower()
    if not met in met_opts:
        error_string = "The distance metric should be one of " + str(met_opts) + " not " + str(met)
        raise ValueError(error_string)

    distances = pdist(df.to_numpy(), metric = met)
    dist_matrix = squareform(distances)
    return(dist_matrix)


def load_association_data(path_dir, eff_fname = "/transformed_eff_dfs.csv",
                                 exp_fname = "/transformed_stdBeta_EXP.csv"):
    
    """Load the beta association scores and standard errors.
    Any transformations for dealing with negative values or normalisation
    should already have been performed on this data.

    Args:
        path_dir (string): Location of the data files.
        eff_fname (string): Name of the csv file for the association scores. default: "/transformed_eff_dfs.csv"
        exp_fname (string): Name of the csv file for the exposure association values. default: "/transformed_stdBeta_EXP.csv"

    eff_df = The association dataframe, rows are snps, columns are traits, cell-values are association values.
    stdBeta_EXP_df = The exposure dataframe, rows are snps, column for the exposure trait, cell-values are association values.
    out = {"eff_df": eff_df, "exp_df": stdBeta_EXP_df}
    Returns:
        out (dict): Dictionary containing the dataframe for the association 
        values and the standard errors.
    """
    # Check all files exist
    if not os.path.exists(path_dir):
        error_string = "Requested directory " + path_dir + " does not exist."
        raise ValueError(error_string)
    if not os.path.exists(path_dir + eff_fname):
        error_string = "Requested file " + eff_fname + " does not exist at " + path_dir
        raise ValueError(error_string)
    if not os.path.exists(path_dir + exp_fname):
        error_string = "Requested file " + exp_fname + " does not exist at " + path_dir
        raise ValueError(error_string)
    
    eff_df = pd.read_csv(path_dir + eff_fname, index_col=0)
    stdBeta_EXP = pd.read_csv(path_dir + exp_fname, index_col=0)

    # Check the exposure column 
    if stdBeta_EXP.shape[1] > 1:
        print("Too many columns included with exposure data. Should be 1. The first column will be used.")
    
    # Drop NaNs
    eff_df = eff_df.dropna(axis = 1)
    stdBeta_EXP = stdBeta_EXP.dropna()

    # Match the snps
    common_snps = eff_df.index.intersection(stdBeta_EXP.index)
    eff_df = eff_df.loc[common_snps,:]
    stdBeta_EXP = stdBeta_EXP.loc[common_snps,:]

    col1 = stdBeta_EXP.columns[0]
    stdBeta_EXP.rename(columns={col1: "EXP"}, inplace=True)
    return {"eff_df" : eff_df, 
            "exp_df" : stdBeta_EXP}

def compute_pca(eff_df, n_components = 2):
    """Compute the principal components from the association data.

    Args:
        eff_df (pd.Dataframe): The association dataframe, rows are snps, columns are traits, cell-values are association values.
        n_components (int): The number of principal components to reduce to.
    Find the two most dominant principal components for the vector space.

    Returns:
        pd_df (pd.Dataframe): Rows are the traits, columns correspond to the principal component.
    """

    # Check input is dataframe exceeding two dimensions
    if not isinstance(eff_df, pd.DataFrame):
        error_string = "The input to `compute_pca` should be a pandas dataframe not " + str(type(eff_df))
        raise TypeError(error_string)
    if not isinstance(n_components, int):
        error_string = "The number of principal components should be an integer not " + str(type(n_components))
    if not eff_df.shape[1] > n_components:
        print("The number of traits does not exceed to number of components so no dimensionality reduction will occur.")
    # Normalise the data
    x = eff_df.values
    #beta_scale = StandardScaler().fit_transform(x)
    beta_scale = StandardScaler().fit_transform(x)
    # Create the column labels
    feat_cols = ['feat_' + str(i) for i in range(beta_scale.shape[1])]
    # Combine the normalised data and the column labels into a dataframe
    norm_beta = pd.DataFrame(beta_scale, columns = feat_cols)
    # Find the first two principal components and return the transformed dataset.
    pca = PCA(n_components = n_components)
    pca_beta = pca.fit_transform(norm_beta)
    pca_df = pd.DataFrame(
        index = eff_df.index,
        data = pca_beta,
        columns = ['pc_%d'%i for i in range(1, n_components + 1)])
    return(pca_df)



