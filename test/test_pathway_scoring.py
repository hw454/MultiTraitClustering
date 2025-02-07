"""
Author: Hayley Wragg
Description: TestPathwayScoring unit tests for the functions in the pathway_scoring module
Type: UnitTests
Created: 5th February 2025
"""

import unittest
import random
import numpy as np
import pandas as pd

from multitraitclustering import pathway_scoring as ps

class TestPathwayScoring(unittest.TestCase):
    """
    TestPathwayScoring unit tests for the functions in the pathway_scoring module
    """
    def test_ssd(self):
        """ssd computes the average of the absolute of the pointwise difference"""
        nc = 6
        a_mat = np.random.rand(nc, nc)
        b_mat = np.eye(nc)
        c_mat = ps.ssd(a_mat, b_mat)
        # Check output is a float
        self.assertTrue(isinstance(c_mat, float))
        # -----------------------
        # NEGATIVE CHECKS
        # TypeError if a_mat is not an np array
        self.assertRaises(TypeError, ps.ssd,
                          a_mat = 3, b_mat = b_mat)
        # TypeError if b_mat is not an np array
        self.assertRaises(TypeError, ps.ssd,
                          a_mat = a_mat, b_mat = 8)
        # ValueError if differing no. rows
        a_mat = np.random.rand(nc, nc + 3)
        b_mat = np.eye(nc + 3)
        self.assertRaises(TypeError, ps.ssd,
                          a_mat = 3, b_mat = b_mat)
        # ValueError if differing no. columns
        a_mat = np.random.rand(nc + 3, nc)
        b_mat = np.eye(nc + 3)
        self.assertRaises(TypeError, ps.ssd,
                          a_mat = 3, b_mat = b_mat)
    def test_uniqueness(self):
        """
        test_uniqueness uniqueness estimate how close to unique columns/rows the data in df is.
        """
        npoints = 300
        nclusts = 6
        pathways = [f"pathway_{i}" for i in range(npoints)]
        cnums = [random.randint(1, nclusts) for i in range(npoints)]
        scores = [random.random() for i in range(npoints)]
        df = pd.DataFrame(data = {"pathway": pathways,
                                  "ClusterNumber": cnums,
                                  "combined_score": scores})
        contain_score = ps.uniqueness(df, axis= 0)
        separation_score = ps.uniqueness(df, axis = 1)
        # Check output is a float
        self.assertTrue(isinstance(contain_score, float))
        self.assertTrue(isinstance(separation_score, float))
        # ------------------------
        # NEGATIVE CHECKS
        # TypeError when df is not a pd DataFrame
        self.assertRaises(TypeError, ps.uniqueness,
                          df = df.to_numpy())
        # TypeError when axis is not an integer
        self.assertRaises(TypeError, ps.uniqueness,
                          df = df,
                          axis = '0')
        # ValueError when score_lab is not a valid column
        self.assertRaises(ValueError, ps.uniqueness,
                          df = df, score_lab = "CombinedScore")
        # ValueError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(ValueError, ps.uniqueness,
                          df = df_no_path)
        # ValueError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(ValueError, ps.uniqueness,
                          df = df_no_clusts)
    def test_assign_max_and_crop(self):
        """
        test_assign_max_and_crop  Find row with max value for cols, crops data for duplicates.
        """
        nr = 302
        nc = 6
        nterms = 50
        a = np.empty((nr, nc))
        a[:] = np.nan
        rows = [[random.randint(0, nr-1) for i in range(nterms)] for j in range(nc)]
        for j in range(nc):
            a[rows[j],j] = [random.random() for i in range(nterms)]
        one_pass = ps.assign_max_and_crop(a)
        # Check output is a dictionary
        self.assertTrue(isinstance(one_pass, dict))
        # Check the types of the terms in the output
        self.assertTrue(isinstance(one_pass["out_mat"], np.ndarray))
        self.assertTrue(isinstance(one_pass["fixed_positions"], list))
        self.assertTrue(isinstance(one_pass["col_pairs"], list))
        # Check for repeated max's
        max_val = np.nan_to_num(a).max()
        a[5,[0,3]] = max_val + 5
        one_pass = ps.assign_max_and_crop(a)
        # Check output is a dictionary
        self.assertTrue(isinstance(one_pass, dict))
        # Check the types of the terms in the output
        self.assertTrue(isinstance(one_pass["out_mat"], np.ndarray))
        self.assertTrue(isinstance(one_pass["fixed_positions"], list))
        self.assertTrue(isinstance(one_pass["col_pairs"], list))
        # -------------------------------
        # NEGATIVE CHECKS
        self.assertRaises(TypeError, ps.assign_max_and_crop, mat = 4)
    def test_overall_paths(self):
        """
        test_overall_paths get an overall pathway score for clusters
        """
        npoints = 300
        nclusts = 6
        pathways = [f"pathway_{i}" for i in range(npoints)]
        cnums = [random.randint(1, nclusts) for i in range(npoints)]
        scores = [random.random() for i in range(npoints)]
        df = pd.DataFrame(data = {"pathway": pathways,
                                  "ClusterNumber": cnums,
                                  "combined_score": scores})
        score = ps.overall_paths(df)
        # Check output is float
        self.assertTrue(isinstance(score, float))
        # ---------------------
        # NEGATIVE CHECKS
        # TypeError if df not dataframe
        self.assertRaises(TypeError, ps.overall_paths, df.to_numpy())
        # ValueError if score_lab is not a valid label
        self.assertRaises(ValueError, ps.overall_paths, df,
                          score_lab = "CombinedScore")
        # ValueError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(ValueError, ps.overall_paths,
                          df = df_no_path)
        # ValueError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(ValueError, ps.overall_paths,
                          df = df_no_clusts)
    def test_redirect_score(self):
        """
        test_redirect_score Makes low values high and vise versa.
        """
        # Check for int input
        self.assertTrue(isinstance(ps.redirect_score(15), float))
        # Check for float input
        self.assertTrue(isinstance(ps.redirect_score(13.2), float))
        # Check for "Nan" input
        self.assertIsNone(ps.redirect_score("NaN"))
        # ---------------------
        # NEGATIVE CHECKS
        # Input is negative
        self.assertRaises(ValueError, ps.redirect_score, score = -4)
        # Input is an invalid string
        self.assertRaises(ValueError, ps.redirect_score, score = "invalid")
        # Input is the wrong type
        self.assertRaises(TypeError, ps.redirect_score, score = np.zeros((3,2)))
    def test_path_best_matches(self):
        """
        test_overall_paths get an overall pathway score for clusters
        """
        npoints = 300
        nclusts = 6
        pathways = [f"pathway_{i}" for i in range(npoints)]
        cnums = [random.randint(1, nclusts) for i in range(npoints)]
        scores = [random.random() for i in range(npoints)]
        df = pd.DataFrame(data = {"pathway": pathways,
                                  "ClusterNumber": cnums,
                                  "combined_score": scores})
        best_out = ps.path_best_matches(df)
        # Check output is dict
        self.assertTrue(isinstance(best_out, dict))
        # Check types of the terms in the dict
        self.assertTrue(isinstance(best_out["best_mat"], np.ndarray))
        self.assertTrue(isinstance(best_out["row_positions"], list))
        self.assertTrue(isinstance(best_out["col_pairs"], list))
        # ---------------------
        # NEGATIVE CHECKS
        # TypeError if df not dataframe
        self.assertRaises(TypeError, ps.path_best_matches, df.to_numpy())
        # ValueError if score_lab is not a valid label
        self.assertRaises(ValueError, ps.path_best_matches, df,
                          score_lab = "CombinedScore")
        # ValueError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(ValueError, ps.path_best_matches,
                          df = df_no_path)
        # ValueError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(ValueError, ps.path_best_matches,
                          df = df_no_clusts)
    def test_clust_path_score(self):
        """Tests the `clust_path_score` function in the pathway_scoring module.
        This test verifies that the clust_path_score function correctly calculates
        the pathway scores for given clusters. It checks the accuracy and correctness
        of the scoring mechanism by comparing the output with expected results.
        Raises:
            AssertionError: If clust_path_Score fails the tests
        """
        npoints = 300
        nclusts = 6
        pathways = [f"pathway_{i}" for i in range(npoints)]
        cnums = [random.randint(1, nclusts) for i in range(npoints)]
        scores = [random.random() for i in range(npoints)]
        df = pd.DataFrame(data = {"pathway": pathways,
                                  "ClusterNumber": cnums,
                                  "combined_score": scores})
        score_out = ps.clust_path_score(df)
        # Check output is dict
        self.assertTrue(isinstance(score_out, dict))
        # Check types of the terms in the dict
        self.assertTrue(isinstance(score_out["PathContaining"], float))
        self.assertTrue(isinstance(score_out["PathSeparating"], float))
        self.assertTrue(isinstance(score_out["OverallPathway"], float))
        # ---------------------
        # NEGATIVE CHECKS
        # TypeError if df not dataframe
        self.assertRaises(TypeError, ps.clust_path_score, df.to_numpy())
        # ValueError if score_lab is not a valid label
        self.assertRaises(ValueError, ps.clust_path_score, df,
                          score_lab = "CombinedScore")
        # ValueError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(ValueError, ps.clust_path_score,
                          df = df_no_path)
        # ValueError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(ValueError, ps.clust_path_score,
                          df = df_no_clusts)
#         ## Test pathway scoring
# * Identity matrix should score minimum.
# * Permuted identity should also score minimum.
# * Identity + Grey squares should score minimum + grey values * number of grey squares.
# * Constant matrix should score maximum.
# nsq=6
# grey_val = 0.5
# n_grey = 2
# # Create the test data
# A0 = grey_val * np.ones((nsq,nsq))
# A4 = np.eye(nsq)
# A1 = A4.copy()
# A1[:,0:3] = A4[:,1:4]
# A1[:,3] = A4[:,0]
# A1[:,4] = A4[:,5]
# A1[:,5] = A4[:,4]
# A2 = A1.copy()
# A2[5,3] = grey_val
# A2[4,4] = grey_val
# def long_df_from_arr(A):
#     nrows = A.shape[0]
#     ncols = A.shape[1]
#     long_dict = {}
#     row = 0
#     for i in range(nrows):
#         for j in range(ncols):
#             one_dict = {"pathway":"p%d"%i,
#                         "ClusterNumber":"c%d"%j,
#                         "combined_score":A[i,j]}
#             long_dict[row] = one_dict
#             row+=1
#     long_df = pd.DataFrame(long_dict).T
#     return(long_df)

# # Set the expected values
# e0 = 1
# e1 = 0
# e2 = grey_val * n_grey / (nsq**2)
# e3 = 0
# # Create a long form data_frame from matrix
# dfs = [long_df_from_arr(A) for A in [A0, A1, A2, A4]]
# expec = [e0, e1, e2, e3]
# test = 0
# for A in dfs:
#     p_scores = ps.clust_path_score(A, score_lab="combined_score")
#     print("Overall score ", p_scores["OverallPathway"])
#     print("expected ", expec[test])
#     test+=1

# nsq=6
# grey_val = 0.5
# # Create the test data
# A3 = np.eye(nsq)
# A3[5,3] = grey_val
# A3[4,4] = grey_val
# A3[5,2] = grey_val
# A3[5,0] = grey_val
# A3[4,3] = grey_val
# A3[4,5] = grey_val
# p_scores = ps.clust_path_score(long_df_from_arr(A3), score_lab="combined_score")
# p_scores

if __name__ == '__main__':
    unittest.main()
