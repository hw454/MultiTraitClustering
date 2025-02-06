"""
Author: Hayley Wragg
Description: Functions for scoring how well clusters correspond to biological pathways
Type: Analysis
Created: 5th February 2025
"""
import numpy as np
import pandas as pd

def ssd(a_mat, b_mat):
    """
    ssd Sum of the absolute difference divided by the number of terms.

    :param a_mat: Array of scores. Rows are pathways, columns are clusters
    :type a_mat: np.ndarray
    :param b_mat: Array of scores. Rows are pathways, columns are clusters
    :type b_mat: np.ndarray
    :return: Absolute value of pointwise difference divided by the number of terms.
    :rtype: float
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
    """
    uniqueness estimate how close to unique columns/rows the data in df is.

    df rows are pathways and columns are clusters. Uniqueness describes how well
    the clusters have identified a unique pathway (pathway containment, axis = 0),
    or how well the clusters have stopped pathways being split across clusters 
    (pathway separation, axis =1).

    :param df: Cluster-Pathway results. rows are pathways and columns are clusters
    :type df: pd.DataFrame
    :param axis: Integer for axis to test on. 0 for rows, 1 for columns, defaults to 0
    :type axis: int, optional
    :param score_lab: col label for data pairing cluster and pathway, defaults "combined_score"
    :type score_lab: str, optional
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
        error_string = f"""column `pathway` should be in df. Available cols: {str(df.columns)}"""
        raise ValueError(error_string)
    if "ClusterNumber" not in df.columns:
        error_string = f"""column `ClusterNumber` should be in df. Available cols: {str(df.columns)}"""
        raise ValueError(error_string)
    # Verify axis is either 0 or 1
    if not (axis == 0 or axis == 1):
        error_string = f"""axis can only be 0 or 1 not {axis}"""
        raise ValueError(error_string)
    # Verify score_lab is a valid column
    if score_lab not in df.columns:
        error_string = f"""score_lab {score_lab} should be a column in df. Available cols: {str(df.columns)}"""
        raise ValueError(error_string)
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
    score = ssd(max_mat, mat_norm)
    return score

def assign_max_and_crop(mat):
    """
    assign_max_and_crop Fix row with max for each col. Crop data left from duplicates

    Fix the rows which are the maximum for only one column.

    Rows which are the maximum for multiple columns are assigned to the column with the
    largest value.

    The matrix is returned with the fixed rows and columns set to zero.

    :param mat: Original data matrix
    :type mat: np.ndarray
    :param out_mat:
    :return: `fixed_positions` - the fixed rows, 
             `col_pairs` - their paired columns,
             `out_mat` - the cropped matrix
    :rtype: dict
    """
    # Verify Types
    if not isinstance(mat, np.ndarray):
        error_string = f"mat should be numpy array not {type(mat)}"
        raise TypeError(error_string)
    out_mat = np.zeros(mat.shape)
    # Initialise - For each column get the row with the highest score
    positions = np.argmax(mat, axis = 0)
    # Are any of the rows repeated?
    unique_ps, counts = np.unique(positions, return_counts = True)
    # Number of unique rows found
    nu = len(unique_ps)
    # All rows with count > 1 need to be assessed
    reassign_ps = [unique_ps[i] for i in range(nu) if counts[i] > 1]
    # All rows with count 1 can be fixed
    fixed_ps = [unique_ps[i] for i in range(nu) if counts[i] == 1]
    # Fix the columns for the fixed rows.
    col_pairs = [positions[positions == p] for p in fixed_ps]
    # Which rows can still be considered?
    # Note this will consider rows fixed on previous iterations but their values
    # will be zero from the cropping so they shouldn't get picked up
    consider_rows = [i for i in range(mat.shape[0]) if i not in fixed_ps]
    # For each row needing reassignment fix the column with the largest value.
    # This will only consider the unassigned columns as the fixed ones will have zero values.
    for p in reassign_ps:
        cols = positions[positions == p]
        max_col = np.argmax(mat[p, cols], axis = 1)
        cols_not_max = cols[cols != max_col]
        consider_rows.remove(p)
        fixed_ps += [p]
        col_pairs += [max_col]
        out_mat[consider_rows, cols_not_max] = mat[consider_rows, cols_not_max]
    out_dict = {"fixed_positions": fixed_ps,
                "col_pairs": col_pairs,
                "out_mat": out_mat}
    return out_dict

# TODO test overall paths function. Redesign to remove nesting. Recursion?
def overall_paths(df, score_lab = "combined_score"):
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

    :param df: columns are: `pathway`, `ClusterNumber` and `combined_score`.
    :type df: pd.DataFrame
    :param score_lab: col label for score, defaults to "combined_score"
    :type score_lab: str, optional
    """
    # Verify Types
    if not isinstance(df, pd.DataFrame):
        error_string = f"""df should be a pandas dataframe not {type(df)}"""
        TypeError(error_string)
    # Verify Columns Labels
    if "pathway" not in df.columns:
        error_string = f"""col `pathway` should be in df. Available cols: {str(df.columns)}"""
        raise ValueError(error_string)
    if "ClusterNumber" not in df.columns:
        error_string = f"""col `ClusterNumber` should be in df. Available cols: {str(df.columns)}"""
        raise ValueError(error_string)
    # Verify score_lab is a valid column
    if score_lab not in df.columns:
        error_string = f"""score_lab {score_lab} not col in df. Available cols: {str(df.columns)}"""
        raise ValueError(error_string)
    df_wide = df.pivot_table(index='pathway', columns='ClusterNumber', values=score_lab)
    mat = np.nan_to_num(df_wide.to_numpy())
    # Get the row number for the maximum in each column
    # For any repeated row numbers get the row number of the second highest
    # Repeat until square matrix (Cropped matrix)
    # Construct matrix of ones in these positions - this is the ideal matrix.
    # Find the sum of the square difference between the cropped matrix and the ideal matrix
    #-----------------
    positions = []
    col_pairs = []
    out_mat = mat.copy()
    for i in range(mat.shape[1]):
        # Max number of iterations is the number of columns
        out_dict = assign_max_and_crop(out_mat)
        positions += out_dict["fixed_positions"]
        col_pairs += out_dict["col_pairs"]
        out_mat = out_mat["out_mat"]
        if len(positions) >= mat.shape[1]:
            break
    crop_mat = mat[positions, :]
    ideal_mat = np.zeros(crop_mat.shape)
    ideal_mat[positions, col_pairs] = mat[positions, col_pairs]
    score = ssd(crop_mat, ideal_mat)
    return(score)

# #### Score shift for making high values good
# # TODO #10 build `redirect` into original pathway score
# def redirect_score(score):
#   if score == "NaN":
#       r_score = None
#   else:
#       r_score = 1/(0.01+score)
#   return(r_score)

# #### The best matches of pathway to clusters
# # TODO #11 test best matches
# def path_best_matches(df, score_lab = "combined_score"):
#     df_wide = df.pivot_table(index='pathway', columns='ClusterNumber', values=score_lab)
#     mat = np.nan_to_num(df_wide.to_numpy())
#     # Get the row number for the maximum in each column
#     # For any repeated row numbers get the row number of the second highest
#     # Repeat until square matrix (Cropped matrix)
#     # Construct matrix of ones in these positions - this is the ideal matrix.
#     # Find the sum of the square difference between the cropped matrix and the ideal matrix
#     positions = np.argmax(mat, axis = 0)
#     uniques, counts = np.unique(positions, return_counts = True)
#     nu = len(uniques)
#     if not len(positions) == nu:
#       for i in range(nu):
#         if counts[i] > 1:
#           row = uniques[i]
#           cols = positions[positions == row]
#           max_col = np.argmax(mat[row, cols])
#           cols_no_max = cols[cols != max_col]
#           j = 2
#           for c in cols_no_max:
#             next_max = np.partition(mat[:, c].flatten(), -j)[-j]
#             row_2 = np.where(mat[:, c] == next_max)
#             if row_2 in positions:
#               j += 1
#             else:
#               break
#             positions[c] = row_2
#     paths = list(df_wide.index[positions])
#     crop_mat = df[df.pathway.isin(paths)].fillna(0)
#     return(crop_mat)

# #### All pathway scores on set of clusters
# # TODO test pathway scoring. Separation score should not be returning so many NaNs
# def clust_path_score(df, score_lab = score_lab):
#   path_contain_score = uniqueness(df, axis = 0, score_lab = score_lab)
#   path_separate_score = uniqueness(df, axis = 1, score_lab = score_lab)
#   path_overall_score = overall_paths(df, score_lab = score_lab)
#   out_dict = {"PathContaining": path_contain_score,
#                "PathSeparating": path_separate_score,
#                "OverallPathway": path_overall_score,
#                "PathContainingRedirected": redirect_score(path_contain_score),
#                "PathSeparatingRedirected": redirect_score(path_separate_score),
#                "OverallPathwayRedirected": redirect_score(path_overall_score)
#   }
#   return(out_dict)