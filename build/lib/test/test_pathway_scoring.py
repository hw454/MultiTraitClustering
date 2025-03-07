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
from multitraitclustering import data_processing as dp

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
        self.assertRaises(ValueError, ps.ssd,
                          a_mat = a_mat, b_mat = b_mat)
        # ValueError if differing no. columns
        a_mat = np.random.rand(nc + 3, nc)
        b_mat = np.eye(nc + 3)
        self.assertRaises(ValueError, ps.ssd,
                          a_mat = a_mat, b_mat = b_mat)
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
        # KeyError when score_lab is not a valid column
        self.assertRaises(KeyError, ps.uniqueness,
                          df = df, score_lab = "CombinedScore")
        # KeyError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(KeyError, ps.uniqueness,
                          df = df_no_path)
        # KeyError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(KeyError, ps.uniqueness,
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
        one_pass = ps.assign_max_and_crop(a, ignore_cols = [])
        # Check output is a dictionary
        self.assertTrue(isinstance(one_pass, dict))
        one_pass = ps.assign_max_and_crop(a, ignore_cols = [2])
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
                                  "CombinedScore": scores})
        best_out = ps.path_best_matches(df)
        # Check output is dict
        self.assertTrue(isinstance(best_out, dict))
        # Check types of the terms in the dict
        self.assertTrue(isinstance(best_out["best_df"], pd.DataFrame))
        self.assertTrue(isinstance(best_out["row_positions"], list))
        self.assertTrue(isinstance(best_out["col_pairs"], list))
        # Check the number of rows equals the number of cols
        self.assertTrue(len(best_out["row_positions"])==len(best_out["col_pairs"]))
        # ---------------------
        # NEGATIVE CHECKS
        # TypeError if df not dataframe
        self.assertRaises(TypeError, ps.path_best_matches, df.to_numpy())
        # KeyError if score_lab is not a valid label
        self.assertRaises(KeyError, ps.path_best_matches, df,
                          score_lab = "invalid_lab")
        # KeyError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(KeyError, ps.path_best_matches,
                          df = df_no_path)
        # KeyError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(KeyError, ps.path_best_matches,
                          df = df_no_clusts)
    def test_clust_path_score(self):
        """Tests the `clust_path_score` function in the pathway_scoring module.
        This test verifies that the clust_path_score function correctly calculates
        the pathway scores for given clusters. It checks the accuracy and correctness
        of the scoring mechanism by comparing the output with expected results.
        Raises:
            AssertionError: If clust_path_score fails the tests
        """
                # Known Value tests
        # Square matrices - nc columns and rows
        # * A0 -identity - Scores 0, then 100 after redirect
        # * A1 -permuted identity - Scores 0, then 100 after redirect
        # * A2 -A1 with two extra smaller values - Scores 2*value/nc**2
        # * A3 - A1 with two extra larger values - Scores 2*smaller value/ nc**2
        # * A4 - Constant matrix - Scores value*(nc-1)/nc
        # Rectangular matrix - npa rows, nc columns
        # * A5 - zeroes with identity inside - Scores 0, then 100 after redirect
        # * A6 - permuted A5 - Scores 0
        # * A7 - A6 with 5 smaller extra values - Score 5*value/(nc*np)
        # * A8 constant - Scores value*(np-1)/np
        nc = 6
        npa = 50
        grey_val = 0.5
        grey_two_val = 0.5* grey_val
        n_grey = 3
        # Create the test data - Square matrix
        a4 = grey_val * np.ones((nc,nc))
        a0 = grey_val * np.eye(nc)
        a1 = a4.copy()
        a1[:,0:3] = a0[:, 1:4]
        a1[:,3] = a0[:, 0]
        a1[:,4] = a0[:, 5]
        a1[:,5] = a0[:, 4]
        a2 = a1.copy()
        a2[5,3] = grey_two_val
        a2[4,4] = grey_two_val
        a2[0,5] = grey_two_val
        a3 = a1.copy()
        a3[5,3] = 0.25 + grey_val
        a3[4,4] = 0.25 + grey_val
        a3[0,5] = 0.25 + grey_val
        # Set the expected values
        e0 = 0
        e1 = 0
        e2 = grey_two_val * n_grey / (nc**2)
        e3 = grey_val * n_grey / (nc**2)
        e4 = grey_val * (nc-1) / nc
        col_lab_dict = {i:f"c{i}" for i in range(a0.shape[1])}
        row_lab_dict = {i:f"p{i}" for i in range(a0.shape[0])}
        # Create a long form data_frame from matrix
        expec = [e0, e1, e2, e3, e4]
        re_expec = [ps.redirect_score(e) for e in expec]
        for i, a_mat in enumerate([a0, a1, a2, a3, a4]):#
            a_df = dp.long_df_from_p_cnum_arr(a_mat,
                                    row_lab_dict=row_lab_dict,
                                    col_lab_dict=col_lab_dict,
                                    score_lab="CombinedScore")
            sc = ps.clust_path_score(a_df)
            self.assertTrue(sc["OverallPathway"]==re_expec[i])
        # Create the test data - Rectangular matrix
        a8 = grey_val * np.ones((npa,nc))
        a5 = np.zeros((npa,nc))
        a5[0:nc,:] = grey_val * np.eye(nc)
        #print(a5)
        a6 = a5.copy()
        a6[:,0:3] = a5[:,1:4]
        a6[:,3] = a5[:,0]
        a6[:,4] = a5[:,5]
        a6[:,5] = a5[:,4]
        a7 = a6.copy()
        a7[5,3] = grey_two_val
        a7[4,4] = grey_two_val
        a7[0,5] = grey_two_val
        a7[20,4] = grey_two_val
        a7[35,5] = grey_two_val
        n_grey = 5
        # Set the expected values
        e5 = 0
        e6 = 0
        e7 = grey_two_val * n_grey / (nc*npa)
        e8 = grey_val * (npa-1) / npa
        col_lab_dict = {i:f"c{i}" for i in range(a5.shape[1])}
        row_lab_dict = {i:f"p{i}" for i in range(a5.shape[0])}
        # Create a long form data_frame from matrix
        expec = [ e5, e6, e7, e8]
        re_expec = [ps.redirect_score(e) for e in expec]
        for i, a_mat in enumerate([a5, a6, a7, a8]):
            a_df = dp.long_df_from_p_cnum_arr(a_mat,
                                    row_lab_dict=row_lab_dict,
                                    col_lab_dict=col_lab_dict,
                                    score_lab="CombinedScore")
            sc = ps.clust_path_score(a_df)
            self.assertTrue(sc["OverallPathway"]==re_expec[i])
        npoints = 300
        nclusts = 6
        pathways = [f"pathway_{i}" for i in range(npoints)]
        cnums = [random.randint(1, nclusts) for i in range(npoints)]
        scores = [random.random() for i in range(npoints)]
        df = pd.DataFrame(data = {"pathway": pathways,
                                  "ClusterNumber": cnums,
                                  "CombinedScore": scores})
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
        # KeyError if score_lab is not a valid label
        self.assertRaises(KeyError, ps.clust_path_score, df,
                          score_lab = "invalid_lab")
        # KeyError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(KeyError, ps.clust_path_score,
                          df = df_no_path)
        # KeyError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(KeyError, ps.clust_path_score,
                          df = df_no_clusts)
    def test_overall_not_cropped_paths(self):
        """
        test_overall_paths get an overall pathway score for clusters
        """
        # Known Value tests
        # Square matrices - nc columns and rows
        # * A0 -identity - Scores 0, then 100 after redirect
        # * A1 -permuted identity - Scores 0, then 100 after redirect
        # * A2 -A1 with two extra smaller values - Scores 2*value/nc**2
        # * A3 - A1 with two extra larger values - Scores 2*smaller value/ nc**2
        # * A4 - Constant matrix - Scores value*(nc-1)/nc
        # Rectangular matrix - npa rows, nc columns
        # * A5 - zeroes with identity inside - Scores 0, then 100 after redirect
        # * A6 - permuted A5 - Scores 0
        # * A7 - A6 with 5 smaller extra values - Score 5*value/(nc*np)
        # * A8 constant - Scores value*(np-1)/np
        nc = 6
        npa = 50
        grey_val = 0.5
        grey_two_val = 0.5* grey_val
        n_grey = 3
        # Create the test data - Square matrix
        a4 = grey_val * np.ones((nc,nc))
        a0 = grey_val * np.eye(nc)
        a1 = a4.copy()
        a1[:,0:3] = a0[:, 1:4]
        a1[:,3] = a0[:, 0]
        a1[:,4] = a0[:, 5]
        a1[:,5] = a0[:, 4]
        a2 = a1.copy()
        a2[5,3] = grey_two_val
        a2[4,4] = grey_two_val
        a2[0,5] = grey_two_val
        a3 = a1.copy()
        a3[5,3] = 0.25 + grey_val
        a3[4,4] = 0.25 + grey_val
        a3[0,5] = 0.25 + grey_val
        # Set the expected values
        e0 = 0
        e1 = 0
        e2 = grey_two_val * n_grey / (nc**2)
        e3 = grey_val * n_grey / (nc**2)
        e4 = grey_val * (nc-1) / nc
        col_lab_dict = {i:f"c{i}" for i in range(a0.shape[1])}
        row_lab_dict = {i:f"p{i}" for i in range(a0.shape[0])}
        # Create a long form data_frame from matrix
        expec = [e0, e1, e2, e3, e4]
        re_expec = [ps.redirect_score(e) for e in expec]
        for i, a_mat in enumerate([a0, a1, a2, a3, a4]):#
            a_df = dp.long_df_from_p_cnum_arr(a_mat,
                                    row_lab_dict=row_lab_dict,
                                    col_lab_dict=col_lab_dict,
                                    score_lab="CombinedScore")
            sc = ps.overall_not_cropped_paths(a_df)
            self.assertTrue(sc==re_expec[i])
        # Create the test data - Rectangular matrix
        a8 = grey_val * np.ones((npa,nc))
        a5 = np.zeros((npa,nc))
        a5[0:nc,:] = grey_val * np.eye(nc)
        #print(a5)
        a6 = a5.copy()
        a6[:,0:3] = a5[:,1:4]
        a6[:,3] = a5[:,0]
        a6[:,4] = a5[:,5]
        a6[:,5] = a5[:,4]
        a7 = a6.copy()
        a7[5,3] = grey_two_val
        a7[4,4] = grey_two_val
        a7[0,5] = grey_two_val
        a7[20,4] = grey_two_val
        a7[35,5] = grey_two_val
        n_grey = 5
        # Set the expected values
        e5 = 0
        e6 = 0
        e7 = grey_two_val * n_grey / (nc*npa)
        e8 = grey_val * (npa-1) / npa
        col_lab_dict = {i:f"c{i}" for i in range(a5.shape[1])}
        row_lab_dict = {i:f"p{i}" for i in range(a5.shape[0])}
        # Create a long form data_frame from matrix
        expec = [ e5, e6, e7, e8]
        re_expec = [ps.redirect_score(e) for e in expec]
        for i, a_mat in enumerate([a5, a6, a7, a8]):
            a_df = dp.long_df_from_p_cnum_arr(a_mat,
                                    row_lab_dict=row_lab_dict,
                                    col_lab_dict=col_lab_dict,
                                    score_lab="CombinedScore")
            sc = ps.overall_not_cropped_paths(a_df)
            self.assertTrue(sc==re_expec[i])
        npoints = 300
        nclusts = 6
        pathways = [f"pathway_{i}" for i in range(npoints)]
        cnums = [random.randint(1, nclusts) for i in range(npoints)]
        scores = [random.random() for i in range(npoints)]
        df = pd.DataFrame(data = {"pathway": pathways,
                                  "ClusterNumber": cnums,
                                  "CombinedScore": scores})
        score = ps.overall_not_cropped_paths(df)
        # Check output is float
        self.assertTrue(isinstance(score, float))
        # Check with score_lab
        score = ps.overall_not_cropped_paths(df, score_lab="CombinedScore")
        # Check output is float
        self.assertTrue(isinstance(score, float))
        # Check that the value for OverallPathway from the clust_paths
        # function is the same as calling the function directly
        score_full_func = ps.clust_path_score(df)
        self.assertTrue(score_full_func["OverallPathway"]==score)
        # ---------------------
        # NEGATIVE CHECKS
        # TypeError if df not dataframe
        self.assertRaises(TypeError, ps.overall_not_cropped_paths, df.to_numpy())
        # KeyError if score_lab is not a valid label
        self.assertRaises(KeyError, ps.overall_not_cropped_paths, df,
                          score_lab = "invalid_lab")
        # KeyError when `pathway` is not a column
        df_no_path = df.rename(columns={'pathway':'paths'})
        self.assertRaises(KeyError, ps.overall_not_cropped_paths,
                          df = df_no_path)
        # KeyError when `ClusterNumber` is not a column
        df_no_clusts = df.rename(columns={'ClusterNumber':'clusts'})
        self.assertRaises(KeyError, ps.overall_not_cropped_paths,
                          df = df_no_clusts)

if __name__ == '__main__':
    unittest.main()
