"""
Author: Hayley Wragg
Description: Functions for scoring how well clusters correspond to biological pathways
Type: Analysis
Created: 5th February 2025
"""

import operator
import numpy as np
import pandas as pd

def ssd(a_mat, b_mat):
    """Compute sum of absolute differences between matrices, normalized by the no. of elements.

    Args:
        a_mat (np.ndarray): first matrix of scores. Rows represent pathways, and columns clusters.
        b_mat (np.ndarray): second matrix of scores, with the same structure as a_mat.

    Returns:
        float: Pointwise sum of absolute difference between matrices, divided by no. of elements.

    Raises:
        TypeError: If either `a_mat` or `b_mat` is not a numpy array.
        ValueError: If `a_mat` and `b_mat` have different shapes.
    """
    # Verify input types
    if not isinstance(a_mat, np.ndarray):
        error_string = f"""a_mat should be a numpy array instead got {type(str(b_mat))}"""
        raise TypeError(error_string)
    if not isinstance(b_mat, np.ndarray):
        error_string = f"""b_mat should be a numpy array instead got {type(str(b_mat))}"""
        raise TypeError(error_string)
    # Check the dimensions are compatible
    if a_mat.shape[0] != b_mat.shape[0]:
        error_string = f"""a_mat and b_mat should have the same number of rows. Instead found
            a_mat has {a_mat.shape[0]} rows and b_mat has {b_mat.shape[0]} rows."""
        raise ValueError(error_string)
    if a_mat.shape[1] != b_mat.shape[1]:
        error_string = f"""a_mat and b_mat should have the same number of columns. Instead found
            a_mat has {a_mat.shape[1]} columns and b_mat has {b_mat.shape[1]} columns."""
        raise ValueError(error_string)
    # Compute ssd
    dif = a_mat.ravel() - b_mat.ravel()

    return abs(dif).sum()/(len(a_mat.ravel()))

def uniqueness(df, axis = 0, score_lab = "combined_score"):
    """Estimates how close to unique columns/rows the data in df is.

    DF rows are pathways and columns are clusters. Uniqueness describes how well
    the clusters have identified a unique pathway (pathway containment, axis = 0),
    or how well the clusters have stopped pathways being split across clusters
    (pathway separation, axis =1).

    Args:
        df (pd.DataFrame): Cluster-Pathway results. Rows are pathways and columns are clusters.
        axis (int, optional): Integer for axis to test on. 0 for rows, 1 for columns. Defaults to 0.
        score_lab (str, optional): Col label for the score. Default "CombinedScore".

    Returns:
        float: score representing uniqueness of cluster pathway pairs.
            Returns "NaN" if axis is 1 and only 1 cluster.
            i.e If only one cluster is found then it's not separated the pathways at all.

    Raises:
        TypeError: `df` not a pandas DataFrame
        TypeError: `axis` not an int
        TypeError: `score_lab` not a str
        ValueError: `pathway` or `ClusterNumber` not columns in `df`
        ValueError: `axis` is not 0 or 1
        KeyError: `score_lab` is not a valid column in `df`.
    """

    # Verify Input Types
    if not isinstance(df, pd.DataFrame):
        error_string = f"""df should be a pandas dataframe instead got {type(str(df))}"""
        raise TypeError(error_string)
    if not isinstance(axis,int):
        error_string = f"""axis should be an integer not {type(str(axis))}"""
        raise TypeError(error_string)
    # Verify Columns Labels
    if "pathway" not in df.columns:
        error_string = f"""col `pathway` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    if "ClusterNumber" not in df.columns:
        error_string = f"""col `ClusterNumber` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    # Verify axis is either 0 or 1
    if not (axis == 0 or axis == 1):
        error_string = f"""axis can only be 0 or 1 not {axis}"""
        raise ValueError(error_string)
    # Verify score_lab is a valid column
    if score_lab not in df.columns:
        error_str = f"""score_lab {score_lab} not col in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_str)
    df_wide = df.pivot_table(index='pathway', columns='ClusterNumber', values=score_lab)
    if df_wide.shape[1]==1 and axis ==1:
        return "NaN"
    mat = np.nan_to_num(df_wide.to_numpy())
    if axis == 1:
        mat = mat.T
    # Shift values to start at zero
    mat_norm = mat - mat.min(0)
    i=0
    # Normalise the range of values
    for m in mat.ptp(1):
        if m != 0:
            mat_norm[i] = mat_norm[i]/m
        i+=1
    max_mat = np.zeros(mat_norm.shape)
    # Create the ideal matrix
    max_mat[np.argmax(mat_norm, axis = 0)] = 1
    # Score the difference between the normalised and ideal matrix
    score = redirect_score(ssd(max_mat, mat_norm))
    return score

def assign_max_and_crop(mat, ignore_cols = None):
    """Assign each column to its maximum row, cropping duplicates.

    For each column in the input matrix, identify the row with the maximum value.
    If row is the max for multiple cols, assign it to the column where its greatest.
    The function then returns the matrix with the fixed rows and columns set to zero.
    This ensures new columns are found during the iteration.

    Args:
        mat (np.ndarray): The input data matrix.

    Returns:
        dict: A dictionary containing the following keys:
            - "fixed_positions" (list): indices of rows that are the best match for a column
            - "col_pairs" (list): indices of columns that pair with `fixed_positions`.
            - "out_mat" (np.ndarray): cropped matrix, with fixed rows and cols set to 0.

    Raises:
        TypeError: If `mat` is not a numpy array.
    """
    # Verify Types
    if ignore_cols is not None and not isinstance(ignore_cols, list):
        error_string = f"ignore_cols should either be None or a list, not {type(ignore_cols)}"
        raise TypeError(error_string)
    if not isinstance(mat, np.ndarray):
        error_string = f"mat should be numpy array not {type(mat)}"
        raise TypeError(error_string)
    if ignore_cols is None:
        ignore_cols = []
    mat = np.nan_to_num(mat)
    out_mat = np.zeros(mat.shape)
    # Initialise - For each column get the row with the highest score
    positions = np.argmax(mat, axis = 0)
    pos_dict = {c: pos for c, pos in enumerate(positions) if c not in ignore_cols}
    # Number of unique rows found
    reassign_ps = []
    fixed_ps = []
    col_pairs = []
    for c in pos_dict.keys():
        count = operator.countOf(pos_dict.values(), pos_dict[c])
        # if position is there multiple times put in reassign
        # If position is there once store in unique and store c in col_pairs
        if count > 1:
            reassign_ps += [pos_dict[c]]
        else:
            fixed_ps += [pos_dict[c]]
            col_pairs += [c]
    reassign_ps = set(reassign_ps)
    # Which rows can still be considered?
    # Note this will consider rows fixed on previous iterations but their values
    # will be zero from the cropping so they shouldn't get picked up
    consider_rows = [i for i in range(mat.shape[0]) if i not in fixed_ps]
    # For each row needing reassignment fix the column with the largest value.
    # This will only consider the unassigned columns as the fixed ones will have zero values.
    for p in reassign_ps:
        cols = np.where(positions == p)[0]
        max_col = cols[np.argmax(mat[p, cols])]
        fixed_ps += [p]
        col_pairs += [max_col]
    consider_cols = [c for c in range(mat.shape[1]) if c not in col_pairs]
    consider_rows = [c for c in range(mat.shape[0]) if c not in fixed_ps]
    for c in consider_cols:
        out_mat[consider_rows, c] = mat[consider_rows, c]
    out_dict = {"fixed_positions": fixed_ps,
                "col_pairs": col_pairs,
                "out_mat": out_mat}
    return out_dict

def overall_paths(df, score_lab = "CombinedScore"):
    """
    overall_paths Score for how well clusters identify pathways
    
    Creates a score that describes how close the pathway cluster scores are
    clusters identifying unique pathways.

    Create the best_matches matrix using `path_best_matches`.

    Create the ideal matrix this is all zeros except for ones on the pathway cluster pairs.
    Only the pathways which appear in a best match somewhere are present in this matrix.

    The score is the ssd between the cropped data and the ideal matrix

    Args:
        df (pd.DataFrame): columns are: `pathway`, `ClusterNumber` and `CombinedScore`.
        score_lab (str, optional): col label for score, defaults to "CombinedScore"
    
    Raise:
        TypeError: df not a pandas dataframe
        KeyError: pathway, ClusterNumber or score_lab not in df columns

    Returns:
        score (float)
    """
    # Verify Types
    if not isinstance(df, pd.DataFrame):
        error_string = f"""df should be a pandas dataframe not {type(df)}"""
        raise TypeError(error_string)
    # Verify Columns Labels
    if "pathway" not in df.columns:
        error_string = f"""col `pathway` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    if "ClusterNumber" not in df.columns:
        error_string = f"""col `ClusterNumber` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    # Verify score_lab is a valid column
    if score_lab not in df.columns:
        error_string = f"""score_lab {score_lab} not col in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    df_wide = df.pivot_table(index='pathway', columns='ClusterNumber', values=score_lab)
    mat = np.nan_to_num(df_wide.to_numpy())
    # Compute the best match matrix and get the corresponding indexes
    best_mat_out= path_best_matches(df, score_lab=score_lab)
    crop_df = best_mat_out["best_df"].pivot_table(index='pathway',
                                                  columns='ClusterNumber',
                                                  values=score_lab)
    crop_mat = np.nan_to_num(crop_df.to_numpy())
    rows = best_mat_out["row_positions"]
    cols = best_mat_out["col_pairs"]
    # Compute the overall score using the best match matrix
    ideal_mat = np.zeros(mat.shape)
    for i,c in enumerate(cols):
        ideal_mat[rows[i], c] = mat[rows[i], c]
    i_mat = ideal_mat[sorted(rows), :]
    score = redirect_score(ssd(crop_mat, i_mat))
    return score

def overall_not_cropped_paths(df, score_lab = "CombinedScore"):
    """
    overall_paths Score for how well clusters identify pathways
    
    Creates a score that describes how close the pathway cluster scores are
    clusters identifying unique pathways.

    Create the best_matches matrix using `path_best_matches`.

    Create the ideal matrix this is all zeros except for ones on the pathway cluster pairs,
    zeroes for all other pathway cluster combinations.

    The score is the ssd between the pathway-cluster data and the ideal matrix

    Args:
        df (pd.DataFrame): columns are: `pathway`, `ClusterNumber` and `CombinedScore`.
        score_lab (str, optional): col label for score, defaults to "CombinedScore"
    
    Raise:
        TypeError: df not a pandas dataframe
        KeyError: pathway, ClusterNumber or score_lab not in df columns

    Returns:
        score (float)
    """
    # Verify Types
    if not isinstance(df, pd.DataFrame):
        error_string = f"""df should be a pandas dataframe not {type(df)}"""
        raise TypeError(error_string)
    # Verify Columns Labels
    if "pathway" not in df.columns:
        error_string = f"""col `pathway` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    if "ClusterNumber" not in df.columns:
        error_string = f"""col `ClusterNumber` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    # Verify score_lab is a valid column
    if score_lab not in df.columns:
        error_string = f"""score_lab {score_lab} not col in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    df_wide = df.pivot_table(index='pathway', columns='ClusterNumber', values=score_lab)
    mat = np.nan_to_num(df_wide.to_numpy())
    # Compute the best match matrix and get the corresponding indexes
    best_mat_out= path_best_matches(df, score_lab=score_lab)
    rows = best_mat_out["row_positions"]
    cols = best_mat_out["col_pairs"]
    # Compute the overall score using the best match matrix
    ideal_mat = np.zeros(mat.shape)
    for i,c in enumerate(cols):
        ideal_mat[rows[i], c] = mat[rows[i], c]
    score = redirect_score(ssd(mat, ideal_mat))
    return score


def redirect_score(score):
    """Score shift for making high values good
    
    Args:
        score (float): score value to be reassigned
        
    Raise:
        TypeError: score not string or float
        
    Returns:
        * None if score == "NaN"
        * 1/(0.01+score) if not """
    if not isinstance(score, (float, int, str)):
        error_string = f"""score should be a float, an int or a str not {type(score)}"""
        raise TypeError(error_string)
    if isinstance(score, str):
        if score != "NaN":
            error_string = f"""when score is a string is should be 'NaN' not {score}"""
            raise ValueError(error_string)
    if not isinstance(score, str):
        if score < 0:
            error_string = "score should be non-negative"
            raise ValueError(error_string)
    if isinstance(score, str):
        r_score = None
    else:
        r_score = 1/(0.01+score)
    return r_score

def path_best_matches(df, score_lab = "CombinedScore"):
    """
    overall_paths Score for how well clusters identify pathways
    
    Creates a score that describes how close the pathway cluster scores are
    clusters identifying unique pathways.

    Clusters matched to their highest scoring pathway. If this pathway is matched elsewhere
    the cluster with the highest score keeps the pathway, the other cluster goes to it's next
    highest path. Once each cluster is assigned a unique pathway the pathway set is fixed.

    Crop the original data to just the scores for the identified pathways. 

    Create the ideal matrix this is all zeros except for ones on the pathway cluster pairs.

    The score is the ssd between the cropped data and the ideal matrix

    Args:
        df (pd.DataFrame): columns are: `pathway`, `ClusterNumber` and `CombinedScore`.
        score_lab (str, optional): col label for score, defaults to "CombinedScore"
    
    Raise:
        TypeError: df not a pandas dataframe
        KeyError: pathway, ClusterNumber of score_labe not in df columns

    Returns:
        out_dict (dict):
        * "best_df" (pd.DataFrame): best matches dataframe
        * "row_positions" (list) the rows for the best matches
        * "col_pairs" (list) the columns paired with the rows
    """
    # Verify Types
    if not isinstance(df, pd.DataFrame):
        error_string = f"""df should be a pandas dataframe not {type(df)}"""
        raise TypeError(error_string)
    # Verify Columns Labels
    if "pathway" not in df.columns:
        error_string = f"""col `pathway` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    if "ClusterNumber" not in df.columns:
        error_string = f"""col `ClusterNumber` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    # Verify score_lab is a valid column
    if score_lab not in df.columns:
        error_string = f"""score_lab {score_lab} not col in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    df_wide = df.pivot_table(index='pathway', columns='ClusterNumber', values=score_lab)
    mat = np.nan_to_num(df_wide.to_numpy())
    # Get the row number for the maximum in each column
    # For any repeated row numbers get the row number of the second highest
    # Repeat until square matrix (Cropped matrix)
    # Construct matrix of ones in these positions - this is the ideal matrix.
    # Find the sum of the square difference between the cropped matrix and the ideal matrix
    #-----------------
    fixed_ps = []
    col_pairs = []
    out_mat = mat.copy()
    for _ in range(mat.shape[1]):
        # Max number of iterations is the number of columns
        out_dict = assign_max_and_crop(out_mat, ignore_cols=col_pairs)
        fixed_ps += out_dict["fixed_positions"]
        col_pairs += out_dict["col_pairs"]
        out_mat = out_dict["out_mat"]
        if len(fixed_ps) >= mat.shape[1]:
            break
    crop_mat = mat[fixed_ps, :]
    crop_df = pd.DataFrame(index=df_wide.index[fixed_ps],
                           data = crop_mat,
                           columns=df_wide.columns[col_pairs])
    best_df = crop_df.melt(value_vars = crop_df.columns,
                           var_name = "ClusterNumber",
                           value_name= score_lab,
                           ignore_index = False
    )
    best_df.reset_index(names = ['pathway'], inplace = True)
    best_dict = {"best_df": best_df,
                 "row_positions": fixed_ps,
                 "col_pairs": col_pairs}
    return best_dict

def clust_path_score(df, score_lab = "CombinedScore"):
    """ Generate the three different pathway scores for a cluster results dataframe.
        Args:
            df (pd.DataFrame): DataFrame containing the cluster results.
            score_lab (str, optional): Label for the score column. Defaults to "combined_score".
        Returns:
            dict: Dictionary containing the pathway scores with keys "PathContaining", 
                  "PathSeparating", and "OverallPathway".
    """
    # Verify Types
    if not isinstance(df, pd.DataFrame):
        error_string = f"""df should be a pandas dataframe not {type(df)}"""
        raise TypeError(error_string)
    # Verify Columns Labels
    if "pathway" not in df.columns:
        error_string = f"""col `pathway` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    if "ClusterNumber" not in df.columns:
        error_string = f"""col `ClusterNumber` should be in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    # Verify score_lab is a valid column
    if score_lab not in df.columns:
        error_string = f"""score_lab {score_lab} not col in df. Available cols: {str(df.columns)}"""
        raise KeyError(error_string)
    path_contain_score = uniqueness(df, axis = 0, score_lab = score_lab)
    path_separate_score = uniqueness(df, axis = 1, score_lab = score_lab)
    path_overall_score = overall_paths(df, score_lab = score_lab)
    path_overall_notcropped_score = overall_not_cropped_paths(df, score_lab = score_lab)
    out_dict = {"PathContaining": path_contain_score,
               "PathSeparating": path_separate_score,
               "OverallPathway": path_overall_score,
               "OverallNotCropped": path_overall_notcropped_score
    }
    return out_dict
