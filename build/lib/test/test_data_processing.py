"""
Author: Hayley Wragg
Created: 6th FEbruary 2025
Description:
    Unit tests for the data processing functions in the multitraitclustering package.
Classes:
    TestDataProcessing: Contains unit tests for various data processing functions.
Methods:
    test_compare_results_list_to_external: Tests comparison of cluster results to external results.
    test_centroid_distance: Tests calculation of distances from each point to the centroid.
    test_overlap_score: Tests calculation of overlap scores between clustering results.
    test_overlap_pairs: Tests finding best matching pairs of clusters between clustering methods.
    test_calc_per_from_comp: Tests calculation of percentage overlap from comparison values.
    test_calc_medoids: Tests finding the medoid for each cluster.
"""

import random as rnd

import unittest
import pandas as pd
import numpy as np

from multitraitclustering import data_setup as ds
from multitraitclustering import data_processing as dp


class TestDataProcessing(unittest.TestCase):
    """
    TestDataProcessing is a unittest.TestCase class that contains several test methods.
    
    Validates the functionality of data processing functions in the dp module. 
    The tests include:
    1. test_compare_results_list_to_external: 
        - Tests the comparison of clustering results to external clustering results.
        - Validates the output type and structure.
        - Includes negative checks for invalid input types and values.
    2. test_centroid_distance: 
        - Tests the calculation of the distance for each point to the centroid.
        - Validates the output type, dimensions, and content.
        - Includes negative checks for invalid input types and dimensions.
    3. test_overlap_score: 
        - Tests the computation of an overlap score from comparison values.
        - Validates the output type.
        - Includes a negative check for invalid input types.
    4. test_overlap_pairs: 
        - Tests the identification of pairs of cluster numbers with the best percentage
        match between two clustering methods.
        - Validates the output type and dimensions.
        - Includes negative checks for invalid input types and labels.
    5. test_calc_per_from_comp: 
        - Tests the computation of the percentage overlap between clusters.
        - Validates the output type.
        - Includes a negative check for invalid input types.
    6. test_calc_medoids: 
        - Tests the identification of the medoid for each cluster.
        - Validates the output type, dimensions, and completeness.
        - Includes negative checks for invalid input types and dimensions.
    """

    def test_compare_results_list_to_external(self):
        """comparison of clustering results to external clustering results."""
        data = ds.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        npnts, _ = data["eff_df"].shape
        nclusts = 4
        external_lab = "compare"
        membership = [rnd.randint(1, nclusts - 1) for j in range(1, npnts + 1)]
        mems_1 = [rnd.randint(1, nclusts - 1) for j in range(1, npnts + 1)]
        mems_2 = [rnd.randint(1, nclusts + 3) for j in range(1, npnts + 1)]
        mems_3 = [rnd.randint(0, nclusts + 50) for j in range(1, npnts + 1)]
        external_df = pd.DataFrame(
            index=data["eff_df"].index,
            data=membership,
            columns=[external_lab]
        )
        clust_df = pd.DataFrame(
            index=data["eff_df"].index,
            data={"method_1": mems_1, "method_2": mems_2, "method_3": mems_3},
        )
        compare_results = dp.compare_results_list_to_external(
            clust_df, external_df, external_lab
        )
        # Check the output is the right type
        self.assertTrue(isinstance(compare_results, dict))
        # Check the dictionary items are the right type
        self.assertTrue(isinstance(compare_results["comp_dfs"], dict))
        self.assertTrue(isinstance(compare_results["cluster_matchings"], dict))
        self.assertTrue(isinstance(compare_results["overlap_scores"], dict))
        # Check the first time in each dictionary is the right type
        comp_key0 = list(compare_results["comp_dfs"].keys())[0]
        comp_df0 = compare_results["comp_dfs"][comp_key0]
        clust_match_key0 = list(compare_results["comp_dfs"].keys())[0]
        clust_match0 = compare_results["cluster_matchings"][clust_match_key0]
        overlap_key0 = list(compare_results["overlap_scores"].keys())[0]
        overlap0 = compare_results["overlap_scores"][overlap_key0]

        self.assertTrue(isinstance(comp_df0, pd.DataFrame))
        self.assertTrue(isinstance(clust_match0, pd.DataFrame))
        self.assertTrue(isinstance(overlap0, float))
        # --------------------------
        # NEGATIVE CHECKS
        # clust_df not dataframe
        self.assertRaises(
            TypeError,
            dp.compare_results_list_to_external,
            clust_df=clust_df.to_numpy(),
            external_df=external_df,
            external_lab=external_lab,
        )
        # external_df not dataframe
        self.assertRaises(
            TypeError,
            dp.compare_results_list_to_external,
            clust_df=clust_df,
            external_df=external_df.to_numpy(),
            external_lab=external_lab,
        )
        # external_lab is not a valid label type
        self.assertRaises(
            TypeError,
            dp.compare_results_list_to_external,
            clust_df=clust_df,
            external_df=external_df,
            external_lab=external_df,
        )
        # external_lab is not a valid column value
        self.assertRaises(
            ValueError,
            dp.compare_results_list_to_external,
            clust_df=clust_df,
            external_df=external_df,
            external_lab="invalid",
        )
        # There are no overlapping indices
        external_df.reset_index(inplace=True)
        self.assertRaises(
            ValueError,
            dp.compare_results_list_to_external,
            clust_df=clust_df,
            external_df=external_df,
            external_lab=external_lab,
        )

    def test_centroid_distance(self):
        """find the distance for each point to the centroid"""
        # Create dummy data for testing
        data = ds.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        npnts, ntrts = data["eff_df"].shape
        nclsts = 4
        cents = [
            [rnd.random() for i in range(1, ntrts + 1)]
            for j in range(1, nclsts + 1)
        ]
        cents_df = pd.DataFrame(
            index=[i for i in range(1, nclsts + 1)],
            data=cents,
            columns=data["eff_df"].columns,
        )
        membership = [rnd.randint(1, nclsts - 1) for j in range(1, npnts + 1)]
        membership_ser = pd.Series(index=data["eff_df"].index, data=membership)
        cent_dist = dp.centroid_distance(
            cents_df, data["eff_df"], membership_ser, metric="euc"
        )
        # Check output type
        self.assertTrue(isinstance(cent_dist, pd.DataFrame))
        # Check dimensions
        self.assertTrue(cent_dist.shape[0] == npnts)
        # Check output contains the column "clust_dist"
        self.assertTrue("clust_dist" in cent_dist.columns)
        # Check output contains the column "clust_dist"
        self.assertTrue("clust_num" in cent_dist.columns)
        # Check the negative cases
        # - TYPES
        # data not a dataframe
        self.assertRaises(
            TypeError,
            dp.centroid_distance,
            cents=cents_df,
            data=data["eff_df"].to_numpy(),
            membership=membership_ser,
            metric="euc",
        )
        # cents not a dataframe
        self.assertRaises(
            TypeError,
            dp.centroid_distance,
            cents=cents_df.to_numpy(),
            data=data["eff_df"],
            membership=membership_ser,
            metric="euc",
        )
        # membership not a series
        self.assertRaises(
            TypeError,
            dp.centroid_distance,
            cents=cents_df,
            data=data["eff_df"],
            membership=membership_ser.to_numpy(),
            metric="euc",
        )
        # metric not a string
        self.assertRaises(
            TypeError,
            dp.centroid_distance,
            cents=cents_df,
            data=data["eff_df"],
            membership=membership_ser,
            metric=4,
        )
        # - Dimensions
        # membership and data need the same number of rows.
        self.assertRaises(
            ValueError,
            dp.centroid_distance,
            cents=cents_df,
            data=data["eff_df"],
            membership=membership_ser.iloc[0: npnts - 5],
            metric="euc",
        )
        # The columns in cents and data match
        self.assertRaises(
            ValueError,
            dp.centroid_distance,
            cents=cents_df.loc[:, cents_df.columns[0: ntrts - 2]],
            data=data["eff_df"],
            membership=membership_ser,
            metric="euc",
        )
        # The values in membership correspond to indices in cents
        membership_val_ser = membership_ser.copy()
        membership_val_ser = membership_val_ser.astype(str)
        membership_val_ser.iloc[0] = "invalid"
        self.assertRaises(
            ValueError,
            dp.centroid_distance,
            cents=cents_df,
            data=data["eff_df"],
            membership=membership_val_ser,
            metric="euc",
        )
        # Invalid metric value
        self.assertRaises(
            ValueError,
            dp.centroid_distance,
            cents=cents_df[cents_df.columns[0: ntrts - 2]],
            data=data["eff_df"],
            membership=membership_ser,
            metric="invalid",
        )

    def test_overlap_score(self):
        """overlap score takes comparison values and computes a score."""
        data = ds.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        ndata = data["eff_df"].shape[0]
        nclust1 = 6
        nclust2 = 4
        df1 = pd.DataFrame(
            index=data["eff_df"].index,
            data={
                "clust_num": [rnd.randint(1, nclust1) for i in range(ndata)],
                "clust_dist": [rnd.random() for j in range(ndata)],
            },
        )
        df2 = pd.DataFrame(
            index=data["eff_df"].index,
            data={
                "clust_num": [rnd.randint(1, nclust2) for i in range(ndata)],
                "clust_dist": [rnd.random() for j in range(ndata)],
            },
        )
        lab1 = "clust_num"
        lab2 = "clust_num"
        comp_df = pd.DataFrame(
            data=ds.compare_df1_to_df2(df1, df2,
                                       lab1, lab2))
        comp_val = dp.overlap_score(comp_df)
        # Check output is a float
        self.assertTrue(isinstance(comp_val, float))
        # Check for TypeError when input is not a dataframe
        self.assertRaises(TypeError, dp.overlap_score, comp_df.to_numpy())

    def test_overlap_pairs(self):
        """Finds the set of pairs of cluster numbers which have the best
        percentage match between two clustering methods."""
        data = ds.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        ndata = data["eff_df"].shape[0]
        nclust1 = 6
        nclust2 = 4
        df1 = pd.DataFrame(
            index=data["eff_df"].index,
            data={
                "clust_num": [rnd.randint(1, nclust1) for i in range(ndata)],
                "clust_dist": [rnd.random() for j in range(ndata)],
            },
        )
        df2 = pd.DataFrame(
            index=data["eff_df"].index,
            data={
                "clust_num": [rnd.randint(1, nclust2) for i in range(ndata)],
                "clust_dist": [rnd.random() for j in range(ndata)],
            },
        )
        lab1 = "clust_num"
        lab2 = "clust_num"
        comp_df = pd.DataFrame(
            data=ds.compare_df1_to_df2(df1, df2,
                                       lab1, lab2))
        comp_pairs = dp.overlap_pairs(comp_df, "method_A", "method_B")
        # Check output is a float
        self.assertTrue(isinstance(comp_pairs, pd.DataFrame))
        # Check output has the same number of rows as the in
        # put
        self.assertTrue(comp_pairs.shape[0] == comp_df.shape[0])
        # Check for TypeError when input is not a dataframe
        self.assertRaises(
            TypeError,
            dp.overlap_pairs,
            comp_percent_df=comp_df.to_numpy(),
            meth_lab="A",
            meth_sec_lab="B",
        )
        # Check for TypeError when input label is not a string
        self.assertRaises(
            TypeError,
            dp.overlap_pairs,
            comp_percent_df=comp_df,
            meth_lab=1,
            meth_sec_lab="B",
        )
        # Check for TypeError when input label is not a string
        self.assertRaises(
            TypeError,
            dp.overlap_pairs,
            comp_percent_df=comp_df,
            meth_lab="A",
            meth_sec_lab=1,
        )

    def test_calc_per_from_comp(self):
        """Take the number of points in an overlap between clusters
        and compute this as a percentage of the number in the
        clusters overall (the intersection divided by the union)."""
        data = ds.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        ndata = data["eff_df"].shape[0]
        nclust1 = 6
        nclust2 = 4
        df1 = pd.DataFrame(
            index=data["eff_df"].index,
            data={
                "clust_num": [rnd.randint(1, nclust1) for i in range(ndata)],
                "clust_dist": [rnd.random() for j in range(ndata)],
            },
        )
        df2 = pd.DataFrame(
            index=data["eff_df"].index,
            data={
                "clust_num": [rnd.randint(1, nclust2) for i in range(ndata)],
                "clust_dist": [rnd.random() for j in range(ndata)],
            },
        )
        lab1 = "clust_num"
        lab2 = "clust_num"
        comp_df = pd.DataFrame(
            data=ds.compare_df1_to_df2(df1, df2,
                                       lab1, lab2))
        comp_per_df = dp.calc_per_from_comp(comp_df)
        # Check the output is a dataframe
        self.assertTrue(isinstance(comp_per_df, pd.DataFrame))
        # Check that a type error is raised if input is not a dataframe
        self.assertRaises(TypeError, dp.calc_per_from_comp, comp_df.to_numpy())

    def test_calc_medoids(self):
        """Find the medoid for each cluster"""
        data = ds.load_association_data(
            path_dir="./data/TestData/",
            eff_fname="unstdBeta_df.csv",
            exp_fname="Beta_EXP.csv",
        )
        npoints, _ = data["eff_df"].shape
        nclusts = 4
        membership = [rnd.randint(1, nclusts) for j in range(1, npoints + 1)]
        membership_ser = pd.Series(index=data["eff_df"].index, data=membership)
        dist_df = pd.DataFrame(
            data=ds.mat_dist(data["eff_df"]),
            index=data["eff_df"].index,
            columns=data["eff_df"].index,
        )
        meds = dp.calc_medoids(
            data=data["eff_df"], data_dist=dist_df, membership=membership_ser
        )
        # Check the output is a dataframe
        self.assertTrue(isinstance(meds, pd.DataFrame))
        # Check the dimensions of the output
        self.assertTrue(meds.shape[0] == nclusts)
        self.assertTrue(meds.shape[1] == data["eff_df"].shape[1])
        # Check that all clusters have a corresponding medoid
        self.assertTrue(all([clust in meds.index
                             for clust in membership_ser.unique()]))
        # -------------------
        # NEGATIVE TESTS
        # TypeError is raised if data is not a dataframe
        self.assertRaises(
            TypeError,
            dp.calc_medoids,
            data=data["eff_df"].to_numpy(),
            data_dist=dist_df,
            membership=membership_ser,
        )
        # TypeError raised if dist_df is not a dataframe
        self.assertRaises(
            TypeError,
            dp.calc_medoids,
            data=data["eff_df"],
            data_dist=dist_df.to_numpy(),
            membership=membership_ser,
        )
        # TypeError raised if membership is not a series
        self.assertRaises(
            TypeError,
            dp.calc_medoids,
            data=data["eff_df"],
            data_dist=dist_df,
            membership=membership_ser.to_numpy(),
        )
        # ValueError raised if data and data_dist have different no. of rows
        self.assertRaises(
            ValueError,
            dp.calc_medoids,
            data=data["eff_df"].iloc[0:-5, :],
            data_dist=dist_df,
            membership=membership_ser,
        )
        # ValueError raised if dist has a differing number of rows to columns
        self.assertRaises(
            ValueError,
            dp.calc_medoids,
            data=data["eff_df"],
            data_dist=dist_df.iloc[0:-5, :],
            membership=membership_ser,
        )
        # ValueError raised if data and membership have different no. of rows
        self.assertRaises(
            ValueError,
            dp.calc_medoids,
            data=data["eff_df"],
            data_dist=dist_df,
            membership=membership_ser.iloc[0:-5],
        )

    def test_long_df_from_p_cnum_arr(self):
        """Test the conversion of a matrix to a long format DataFrame."""
        # Create a sample matrix and label dictionaries
        a_mat = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        row_lab_dict = {0: "pathway_1", 1: "pathway_2"}
        col_lab_dict = {0: "cluster_1", 1: "cluster_2", 2: "cluster_3"}
        score_lab = "score"
        # Call the function
        long_df = dp.long_df_from_p_cnum_arr(
            a_mat, row_lab_dict, col_lab_dict, score_lab
        )

        # Assert that the output is a DataFrame
        self.assertIsInstance(long_df, pd.DataFrame)

        # Assert that the DataFrame has the correct columns
        expected_columns = ["pathway", "ClusterNumber", "score"]
        self.assertListEqual(list(long_df.columns), expected_columns)

        # Assert that the DataFrame has the correct number of rows
        self.assertEqual(long_df.shape[0], 6)
if __name__ == "__main__":
    unittest.main()
