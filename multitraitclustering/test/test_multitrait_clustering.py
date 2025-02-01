import unittest
import pandas as pd
import random as rnd
import numpy as np

import data_manipulation.data_setup as dm
import data_manipulation.data_processing as dp
import clustering.multi_trait_clustering as mtc

class TestMultiTraitClustering(unittest.TestCase):

    def test_method_string(self):
        """The function method_string should take in four inputs (3 string, 1 numeric) and output a string label.
        The output should contain no spaces, underscores or numerics and be in title case.

        """
        meth_str = "method"
        alg_str = "alg"
        dis_str = "dist"
        num = 8
        exclude_list = [ " ", "_"]
        meth_str = mtc.method_string(meth_str, alg_str, dis_str, num)
        exclude_check = lambda s: False if any([l in exclude_list or l.isnumeric()] for l in s) else True
        # Check types
        self.assertTrue(isinstance(meth_str, str))
        # Check exclusions
        self.assertFalse(exclude_check(meth_str))
        # Check the negative cases
        # Check TypeError for the string inputs not as strings
        self.assertRaises(TypeError, mtc.method_string,
                          meth_str = 0,
                          alg_str = alg_str,
                          dis_str = dis_str,
                          num = num)
        self.assertRaises(TypeError, mtc.method_string,
                          meth_str = meth_str,
                          alg_str = 0,
                          dis_str = dis_str,
                          num = num)
        self.assertRaises(TypeError, mtc.method_string,
                          meth_str = meth_str,
                          alg_str = alg_str,
                          dis_str = 0,
                          num = num)
        # Check for TypeError for num not numeric
        self.assertRaises(TypeError, mtc.method_string,
                          meth_str = meth_str,
                          alg_str = alg_str,
                          dis_str = dis_str,
                          num = "num")
    def test_get_aic(self):
        """The function `get_aic` should take in a dataframe with columns `clust_num` and `clust_dist`
        and compute the AIC value as a float."""
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        cnums = 6
        rand_nums = [rnd.randint(0, cnums) for i in range(0, ndata)]
        rand_dist = [rnd.random() for i in range(0, ndata)]
        dummy_clusts = pd.DataFrame(index = data["eff_df"].index,
                                    data = {"clust_num": rand_nums, 
                                            "clust_dist": rand_dist})
        aic_val = mtc.get_aic(dummy_clusts)
        # Check that the AIC is a float
        self.assertTrue(isinstance(aic_val, float))
        # Check that a Type Error is returned if the input is not a data frame.
        self.assertRaises(TypeError, mtc.get_aic, dummy_clusts.to_numpy())
        # Check that a Type Error is returned if a not int dimension is input
        self.assertRaises(TypeError, mtc.get_aic, 
                          clust_mem_dist = dummy_clusts,
                          dimension = "dim")
        # Check that a ValueError is returned is the dimension is less than 2.
        self.assertRaises(ValueError, mtc.get_aic,
                          clust_mem_dist = dummy_clusts,
                          dimension = 1)
        # Check that a ValueError is returned if clust_mem_dist does not have column "clust_num"
        self.assertRaises(ValueError,mtc.get_aic,
                          clust_mem_dist = dummy_clusts.loc[:, dummy_clusts.columns != "clust_num"],
                          dimension = 4)
        # Check that a ValueError is returned if clust_mem_dist does not have column "clust_dist"
        self.assertRaises(ValueError, mtc.get_aic,
                          clust_mem_dist = dummy_clusts.loc[:, dummy_clusts.columns != "clust_dist"],
                          dimension = 4)
    def test_get_bic(self):
        """The function `get_bic` should take in a dataframe with columns `clust_num` and `clust_dist`
        and compute the AIC value as a float."""
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        cnums = 6
        rand_nums = [rnd.randint(0, cnums) for i in range(0, ndata)]
        rand_dist = [rnd.random() for i in range(0, ndata)]
        dummy_clusts = pd.DataFrame(index = data["eff_df"].index,
                                    data = {"clust_num": rand_nums, 
                                            "clust_dist": rand_dist})
        # With default dimension and n_params
        aic_val = mtc.get_bic(dummy_clusts)
        # With larger dimension and n_params
        aic_full_val = mtc.get_bic(dummy_clusts, dimension = 6, n_params = 20)
        # TESTS
        # Check that the AIC is a float
        self.assertTrue(isinstance(aic_val, float))
        # Check that the AIC is a float
        self.assertTrue(isinstance(aic_full_val, float))
        # Check that a Type Error is returned if the input is not a data frame.
        self.assertRaises(TypeError, mtc.get_bic, dummy_clusts.to_numpy())
        # Check that a Type Error is returned if a not int dimension is input
        self.assertRaises(TypeError, mtc.get_bic, 
                          clust_mem_dist = dummy_clusts,
                          dimension = "dim")
        # Check that a Type Error is returned if a not int nparams is input
        self.assertRaises(TypeError, mtc.get_bic, 
                          clust_mem_dist = dummy_clusts,
                          n_params = "pars")
        # Check that a ValueError is returned is the dimension is less than 2.
        self.assertRaises(ValueError, mtc.get_bic,
                          clust_mem_dist = dummy_clusts,
                          dimension = 1)
        # Check that a ValueError is returned if clust_mem_dist does not have column "clust_num"
        self.assertRaises(ValueError,mtc.get_bic,
                          clust_mem_dist = dummy_clusts.loc[:, dummy_clusts.columns != "clust_dist"],
                          dimension = 4)
        # Check that a ValueError is returned if clust_mem_dist does not have column "clust_dist"
        self.assertRaises(ValueError, mtc.get_bic,
                          clust_mem_dist = dummy_clusts.loc[:, dummy_clusts.columns != "clust_dist"],
                          dimension = 4)
    def test_cluster_all_methods(self):
        data = dm.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        ntraits = 112
        rand_data = np.random.rand(ndata, ntraits)
        trait_labs = ["trait%d"%i for i in range(ntraits)]
        dummy_assoc_df = pd.DataFrame(index = data["eff_df"].index,
                                    data = rand_data,
                                    columns = trait_labs)
        clust_outs = mtc.cluster_all_methods(exp_df = data["exp_df"], assoc_df = dummy_assoc_df)
        # Check output is a dictionary with a dataframe and dictionary
        self.assertTrue(isinstance(clust_outs, dict))
        self.assertTrue(isinstance(clust_outs["clust_pars_dict"], dict))
        self.assertTrue(isinstance(clust_outs["clust_results"], pd.DataFrame))
        # Check TypeErrors raised when input is not dataframe
        self.assertRaises(TypeError, mtc.cluster_all_methods, 
                          exp_df = data["exp_df"].to_numpy(), assoc_df = dummy_assoc_df)
        # Check TypeError is raised when assoc is not a dataframe
        self.assertRaises(TypeError, mtc.cluster_all_methods, 
                          exp_df = data["exp_df"], assoc_df = rand_data)
        # Check ValueError is raised when the number of dimensions doesn't match
        self.assertRaises(ValueError, mtc.cluster_all_methods, 
                          exp_df = data["exp_df"], assoc_df = dummy_assoc_df.iloc[0:-2,:])
if __name__ == '__main__':
    unittest.main()