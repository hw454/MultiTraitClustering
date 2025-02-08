"""
Author: Hayley Wragg
Created: 10th December 2024
Description:
    Unit tests for the multi-trait clustering module.

    This module contains several test cases for functions in the
    `multitraitclustering` module, including:

    - `method_string`: Tests the generation of a method string.
    - `get_aic`: Tests the calculation of the Akaike Information Criterion (AIC).
    - `get_bic`: Tests the calculation of the Bayesian Information Criterion (BIC).
    - `cluster_all_methods`: Tests the clustering of all methods.
"""

import unittest
import random as rnd

import pandas as pd
import numpy as np

from multitraitclustering import data_setup as dm
from multitraitclustering import multi_trait_clustering as mtc


def exclude_check(s):
    """
    Checks if a string contains any characters from the exclude_list or any numeric characters.
    Args:
        s (str): The string to check.
    Returns:
        bool: False if string contains characters from exclude_list or numeric, True otherwise.
    """
    exclude_list = [" ", "_"]
    if any([letter in exclude_list or letter.isnumeric()] for letter in s):
        return False
    else:
        return True

class TestMultiTraitClustering(unittest.TestCase):
    """Unit tests for the MultiTraitClustering module.
    This class contains several test methods to verify the functionality of the
    MultiTraitClustering module, including:
    - `test_method_string`: Tests the `method_string` function, which generates a
        formatted string from input strings and a number.  It checks for correct
        output type, content, and proper handling of invalid input types.
    - `test_get_aic`: Tests the `get_aic` function, which calculates the Akaike
        Information Criterion (AIC) from a DataFrame containing cluster assignments
        and distances. It validates the output type and checks for correct error
        handling when the input DataFrame is malformed or of the wrong type.
    - `test_get_bic`: Tests the `get_bic` function, which calculates the Bayesian
        Information Criterion (BIC) from a DataFrame containing cluster assignments
        and distances. It validates the output type and checks for correct error
        handling when the input DataFrame is malformed or of the wrong type, or
        when dimension or n_params are invalid.
    - `test_cluster_all_methods`: Tests the `cluster_all_methods` function, which
        performs clustering using multiple methods. It verifies that the output is
        a dictionary containing a DataFrame of clustering results and a dictionary
        of clustering parameters. It also checks for correct error handling when
        input DataFrames are malformed or of the wrong type, or when the number
        of dimensions doesn't match between dataframes.
"""
    def test_method_string(self):
        """take in four inputs (3 string, 1 numeric) and output a string label.
        output contains no spaces, underscores or numerics and be title case.
        """
        meth_str = "method"
        alg_str = "alg"
        dis_str = "dist"
        num = 8
        meth_str = mtc.method_string(meth_str, alg_str, dis_str, num)
        # Check types
        self.assertTrue(isinstance(meth_str, str))
        # Check exclusions
        self.assertFalse(exclude_check(meth_str))
        # -----------------
        # NEGATIVE CHECKS
        # TypeError for the string inputs not as strings
        self.assertRaises(
            TypeError,
            mtc.method_string,
            meth_str=0,
            alg_str=alg_str,
            dis_str=dis_str,
            num=num,
        )
        self.assertRaises(
            TypeError,
            mtc.method_string,
            meth_str=meth_str,
            alg_str=0,
            dis_str=dis_str,
            num=num,
        )
        self.assertRaises(
            TypeError,
            mtc.method_string,
            meth_str=meth_str,
            alg_str=alg_str,
            dis_str=0,
            num=num,
        )
        # TypeError for num not numeric
        self.assertRaises(
            TypeError,
            mtc.method_string,
            meth_str=meth_str,
            alg_str=alg_str,
            dis_str=dis_str,
            num="num",
        )

    def test_get_aic(self):
        """take in a dataframe with columns `clust_num` and `clust_dist`
        and compute the AIC value as a float."""
        data = dm.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        ndata = data["eff_df"].shape[0]
        cnums = 6
        rand_nums = [rnd.randint(0, cnums) for i in range(0, ndata)]
        rand_dist = [rnd.random() for i in range(0, ndata)]
        dummy_clusts = pd.DataFrame(
            index=data["eff_df"].index,
            data={"clust_num": rand_nums, "clust_dist": rand_dist},
        )
        aic_val = mtc.get_aic(dummy_clusts)
        # Check that the AIC is a float
        self.assertTrue(isinstance(aic_val, float))
        # ---------------
        # NEGATIVE CHECKS
        # Check that a Type Error is returned if the input is not a data frame.
        self.assertRaises(TypeError, mtc.get_aic, dummy_clusts.to_numpy())
        # Check that a Type Error is returned if a not int dimension is input
        self.assertRaises(
            TypeError, mtc.get_aic,
            clust_mem_dist=dummy_clusts,
            dimension="dim"
        )
        # ValueError if the dimension is less than 2.
        self.assertRaises(
            ValueError, mtc.get_aic, clust_mem_dist=dummy_clusts, dimension=1
        )
        # ValueError if clust_mem_dist does not have column "clust_num"
        col_list = dummy_clusts.columns != "clust_num"
        self.assertRaises(
            ValueError,
            mtc.get_aic,
            clust_mem_dist=dummy_clusts.loc[:, col_list],
            dimension=4,
        )
        # ValueError if clust_mem_dist does not have column "clust_dist"
        col_list = dummy_clusts.columns != "clust_dist"
        self.assertRaises(
            ValueError,
            mtc.get_aic,
            clust_mem_dist=dummy_clusts.loc[:, col_list],
            dimension=4,
        )

    def test_get_bic(self):
        """Take in a dataframe with columns `clust_num` and `clust_dist`
        and compute the AIC value as a float."""
        data = dm.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        ndata = data["eff_df"].shape[0]
        cnums = 6
        rand_nums = [rnd.randint(0, cnums) for i in range(0, ndata)]
        rand_dist = [rnd.random() for i in range(0, ndata)]
        dummy_clusts = pd.DataFrame(
            index=data["eff_df"].index,
            data={"clust_num": rand_nums, "clust_dist": rand_dist},
        )
        # With default dimension and n_params
        aic_val = mtc.get_bic(dummy_clusts)
        # With larger dimension and n_params
        aic_full_val = mtc.get_bic(dummy_clusts, dimension=6, n_params=20)
        # TESTS
        # Check that the AIC is a float
        self.assertTrue(isinstance(aic_val, float))
        # Check that the AIC is a float
        self.assertTrue(isinstance(aic_full_val, float))
        # ----------------
        # NEGATIVE CHECKS
        # TypeError if the input is not a data frame.
        self.assertRaises(TypeError, mtc.get_bic, dummy_clusts.to_numpy())
        # TypeError if a not int dimension is input
        self.assertRaises(
            TypeError, mtc.get_bic,
            clust_mem_dist=dummy_clusts,
            dimension="dim"
        )
        # TypeError if a not int nparams is input
        self.assertRaises(
            TypeError, mtc.get_bic,
            clust_mem_dist=dummy_clusts,
            n_params="pars"
        )
        # ValueError if dimension is less than 2.
        self.assertRaises(
            ValueError, mtc.get_bic, clust_mem_dist=dummy_clusts, dimension=1
        )
        # ValueError if clust_mem_dist does not have column "clust_num"
        col_list = dummy_clusts.columns != "clust_dist"
        self.assertRaises(
            ValueError,
            mtc.get_bic,
            clust_mem_dist=dummy_clusts.loc[:, col_list],
            dimension=4,
        )
        # ValueError if clust_mem_dist does not have column "clust_dist"
        col_list = dummy_clusts.columns != "clust_dist"
        self.assertRaises(
            ValueError,
            mtc.get_bic,
            clust_mem_dist=dummy_clusts.loc[:, col_list],
            dimension=4,
        )

    def test_cluster_all_methods(self):
        """
        Test the cluster_all_methods function.
        This test checks the following:
            - The output is a dictionary with a dataframe and dictionary.
            - The keys for the clust_pars_dict match the columns for clust_df.
            - TypeErrors are raised when the input is not a dataframe.
            - ValueError is raised when the number of dimensions doesn't match.
            - KeyError is raised when the exp_df doesn't contain a column named EXP.
        """

        data = dm.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        ndata = data["eff_df"].shape[0]
        ntraits = 112
        rand_data = np.random.rand(ndata, ntraits)
        trait_labs = [f"trait{i}" for i in range(ntraits)]
        dummy_assoc_df = pd.DataFrame(
            index=data["eff_df"].index, data=rand_data, columns=trait_labs
        )
        clust_outs = mtc.cluster_all_methods(
            exp_df=data["exp_df"], assoc_df=dummy_assoc_df
        )
        # Check output is a dictionary with a dataframe and dictionary
        self.assertTrue(isinstance(clust_outs, dict))
        self.assertTrue(isinstance(clust_outs["clust_pars_dict"], dict))
        self.assertTrue(isinstance(clust_outs["clust_results"], pd.DataFrame))
        # Check the keys for the clust_pars_dict match the columns for clust_df
        self.assertTrue(
            set(clust_outs["clust_pars_dict"].keys())
            == set(clust_outs["clust_results"].columns)
        )
        # ----------------
        # NEGATIVE CHECKS
        # TypeErrors when input is not dataframe
        self.assertRaises(
            TypeError,
            mtc.cluster_all_methods,
            exp_df=data["exp_df"].to_numpy(),
            assoc_df=dummy_assoc_df,
        )
        # TypeError when assoc is not a dataframe
        self.assertRaises(
            TypeError,
            mtc.cluster_all_methods,
            exp_df=data["exp_df"],
            assoc_df=rand_data,
        )
        # ValueError when the number of dimensions doesn't match
        self.assertRaises(
            ValueError,
            mtc.cluster_all_methods,
            exp_df=data["exp_df"],
            assoc_df=dummy_assoc_df.iloc[0:-2, :],
        )
        # KeyError when the exp_df doesn't contain a column named EXP
        exp_wrong_col = data["exp_df"].rename(
            columns = {data["exp_df"].columns[0]: "invalid"}
        )
        self.assertRaises(
            KeyError,
            mtc.cluster_all_methods,
            exp_df=exp_wrong_col,
            assoc_df=data["eff_df"],
        )

if __name__ == "__main__":
    unittest.main()
