import unittest
import pandas as pd
import random as rnd
import numpy as np

import data_manipulation.data_setup as dm

class TestDataSetup(unittest.TestCase):

    def test_load_association_data(self):
        """Test the data types and non-zero dimensions from loading the association data.
        """
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                                eff_fname = "unstdBeta_df.csv",
                                                exp_fname = "Beta_EXP.csv")
        # Check types
        self.assertTrue(isinstance(data, dict))
        self.assertTrue(isinstance(data["eff_df"], pd.DataFrame))
        self.assertTrue(isinstance(data["exp_df"], pd.DataFrame))
        # Check dimensions
        self.assertTrue(data["eff_df"].shape[0]>0)
        self.assertTrue(data["eff_df"].shape[1]>0)
        self.assertTrue(data["exp_df"].shape[0]>0)
        self.assertTrue(data["exp_df"].shape[1]>0)
        # Check the negative cases
        # Missing SE file
        self.assertRaises(ValueError, dm.load_association_data, path_dir = "../data/TestData/",
                                        eff_fname = "no_name.csv")
        # No file extension
        self.assertRaises(ValueError, dm.load_association_data, path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df",
                                        exp_fname = "Beta_EXP.csv")
        # Incorrect directory location
        self.assertRaises(ValueError, dm.load_association_data, path_dir = "../TestData/",
                                        eff_fname = "unstdBeta_df",
                                        exp_fname = "Beta_EXP.csv")
    def test_compute_pca(self):
        """ Check the compute_pca creates a Dataframe"""
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        pca = dm.compute_pca(data["eff_df"])
        # Check return type is Dataframe
        self.assertTrue(isinstance(pca, pd.DataFrame))
        # Check the function fails when given an array instead of a Dataframe
        self.assertRaises(TypeError, dm.compute_pca, data["eff_df"].to_numpy())
    def test_calc_sum_sq(self):
        """Check the function for calculating the sum of the square of the 
        distances for the points in a given cluster. 
        Positive tests:
        
        * output type is float
        
        Negative tests:
        
        * invalid cluster number raises ValueError
        * data input not a Dataframe raises TypeError
        * missing column from data raises ValueError
        """
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        cnums = 6
        cn = rnd.randint(0, cnums)
        rand_nums = [rnd.randint(0, cnums) for i in range(0, ndata)]
        rand_dist = [rnd.random() for i in range(0, ndata)]
        dummy_clusts = pd.DataFrame(index = data["eff_df"].index,
                                    data = {"clust_num": rand_nums, 
                                            "clust_dist": rand_dist})
        sum_sqs = dm.calc_sum_sq(cn, dummy_clusts)
        # Check the output is a float
        self.assertTrue(isinstance(sum_sqs, float))
        # Check failure for invalid cluster number
        self.assertRaises(IndexError, dm.calc_sum_sq, data = dummy_clusts, c_num = cnums + 1)
        # Check failure for missing data column
        dummy_clusts_missing = pd.DataFrame(index = data["eff_df"].index,
                                    data = {"clust_num": rand_nums})
        self.assertRaises(ValueError, dm.calc_sum_sq, data = dummy_clusts_missing, c_num = cnums - 1 )
        # Check failure for input not as Dataframe   
        self.assertRaises(TypeError, dm.calc_sum_sq, data = dummy_clusts.to_numpy(), c_num = cnums - 1) 
    def test_dist_met(self):
        """Check the function for calculating the distance between all points in a dataframe.
        Positive Tests:
        * Output is a numpy array
        Negative Tests:
        * If input is not a Dataframe raises TypeError
        * If metric is not a string raises TypeError
        * If metric is not a value option raises ValueError
        """
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        met = "Euclidean"
        dist_met = dm.mat_dist(data["eff_df"], met)
        # Check output type
        self.assertTrue(isinstance(dist_met, np.ndarray))
        # Check if data is not a Dataframe raises TypeError
        self.assertRaises(TypeError, dm.mat_dist, data["eff_df"].to_numpy(), met = met)
        # Check if met is not a string raises TypeError
        self.assertRaises(TypeError, dm.mat_dist, data["eff_df"], met = 5)
        # Check if input is not a Dataframe raises TypeError
        self.assertRaises(ValueError, dm.mat_dist, data["eff_df"], met = "not_metric")
    def test_compare_df1_to_df2(self):
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        cnums = 6
        rand_nums1 = [rnd.randint(0, cnums) for i in range(0, ndata)]
        rand_nums2 = [rnd.randint(0, 10 * cnums) for i in range(0, ndata)]
        lab1 = "clust1"
        lab2 = "clust2"
        df1 = pd.DataFrame(index = data["eff_df"].index,
                                    data = {lab1: rand_nums1})
        cnum1 = len(set(rand_nums1))
        df2 = pd.DataFrame(index = data["eff_df"].index,
                                    data = {lab2: rand_nums2})
        cnum2 = len(set(rand_nums2))
        comp = dm.compare_df1_to_df2(clust_df1 = df1, clust_df2 = df2, lab1 = lab1, lab2 = lab2)
        # Check output is an array
        self.assertTrue(isinstance(comp, np.ndarray))
        # Check the dimensions of the output
        self.assertTrue(comp.shape[0] == cnum1)
        self.assertTrue(comp.shape[1] == cnum2)
        # Check that TypeError is raised if either data input is not a Dataframe
        self.assertRaises(TypeError, dm.compare_df1_to_df2, 
                          clust_df1 = df1.to_numpy(), clust_df2 = df2, lab1 = lab1, lab2 = lab2)
        self.assertRaises(TypeError, dm.compare_df1_to_df2, 
                          clust_df1 = df1, clust_df2 = df2.to_numpy(), lab1 = lab1, lab2 = lab2)
        self.assertRaises(TypeError, dm.compare_df1_to_df2, 
                          clust_df1 = df1.to_numpy(), clust_df2 = df2.to_numpy(), 
                          lab1 = lab1, lab2 = lab2)
        # Check that TypeError is raised if the label is not a string
        self.assertRaises(TypeError, dm.compare_df1_to_df2, 
                          clust_df1 = df1, clust_df2 = df2, lab1 = 3, lab2 = lab2)
        self.assertRaises(TypeError, dm.compare_df1_to_df2, 
                          clust_df1 = df1, clust_df2 = df2, lab1 = lab1, lab2 = 3)
        self.assertRaises(TypeError, dm.compare_df1_to_df2, 
                          clust_df1 = df1, clust_df2 = df2, 
                          lab1 = 3, lab2 = 5)
        # Check that ValueError is raised if the label doesn't match a column name
        self.assertRaises(ValueError, dm.compare_df1_to_df2, 
                          clust_df1 = df1, clust_df2 = df2, lab1 = "no_match", lab2 = lab2)
        self.assertRaises(ValueError, dm.compare_df1_to_df2, 
                          clust_df1 = df1, clust_df2 = df2, lab1 = lab1, lab2 = "no_match")
        self.assertRaises(ValueError, dm.compare_df1_to_df2, 
                          clust_df1 = df1, clust_df2 = df2, 
                          lab1 = "no_match", lab2 = "no_match")
if __name__ == '__main__':
    unittest.main()