import unittest
import pandas as pd
import random as rnd

from multitraitclustering import data_setup as ds
from multitraitclustering import data_processing as dp

class TestDataProcessing(unittest.TestCase):

    def test_centroid_distance(self):
        """Test the data types and non-zero dimensions from loading the association data.
        """
        # Create dummy data for testing
        data = ds.load_association_data(path_dir = "./data/TestData/",
                                                eff_fname = "unstdBeta_df.csv",
                                                exp_fname = "Beta_EXP.csv")
        npoints, ntraits = data["eff_df"].shape
        nclusts = 4
        cents = [[rnd.random() for i in range(1, ntraits + 1)] for j in range(1, nclusts + 1)]
        cents_df = pd.DataFrame(index = [ i for i in range(1, nclusts + 1)],
                                data = cents,
                                columns = data["eff_df"].columns)
        membership = [rnd.randint(1, nclusts - 1) for j in range(1, npoints + 1)]
        membership_ser = pd.Series(index = data["eff_df"].index,
                                   data = membership)
        cent_dist = dp.centroid_distance(cents_df, data["eff_df"], membership_ser, metric = "euc")
        # Check types
        self.assertTrue(isinstance(cent_dist, pd.DataFrame))
        # Check dimensions
        self.assertTrue(cent_dist.shape[0] == npoints)
        # Check output contains the column "clust_dist"
        self.assertTrue("clust_dist" in cent_dist.columns)
        # Check output contains the column "clust_dist"
        self.assertTrue("clust_num" in cent_dist.columns)
        # Check the negative cases
        # - TYPES
        # data not a dataframe
        self.assertRaises(TypeError, dp.centroid_distance, 
                          cents = cents_df, 
                          data = data["eff_df"].to_numpy(), 
                          membership = membership_ser, 
                          metric = "euc")
        # cents not a dataframe
        self.assertRaises(TypeError, dp.centroid_distance, 
                          cents = cents_df.to_numpy(), 
                          data = data["eff_df"], 
                          membership = membership_ser, 
                          metric = "euc")
        # membership not a series
        self.assertRaises(TypeError, dp.centroid_distance, 
                          cents = cents_df, 
                          data = data["eff_df"], 
                          membership = membership_ser.to_numpy(), 
                          metric = "euc")
        # metric not a string
        self.assertRaises(TypeError, dp.centroid_distance, 
                          cents = cents_df, 
                          data = data["eff_df"], 
                          membership = membership_ser, 
                          metric = 4)
        # - Dimensions
        # membership and data need the same number of rows.
        self.assertRaises(ValueError, dp.centroid_distance, 
                          cents = cents_df, 
                          data = data["eff_df"], 
                          membership = membership_ser.iloc[0:npoints-5], 
                          metric = "euc")
        # The columns in cents and data match
        self.assertRaises(ValueError, dp.centroid_distance, 
                          cents = cents_df.loc[:, cents_df.columns[0:ntraits-2]], 
                          data = data["eff_df"], 
                          membership = membership_ser, 
                          metric = "euc")
        # The values in membership correspond to indices in cents
        membership_val_ser = membership_ser.copy()
        membership_val_ser = membership_val_ser.astype(str)
        membership_val_ser.iloc[0] = "invalid"
        self.assertRaises(ValueError, dp.centroid_distance, 
                          cents = cents_df, 
                          data = data["eff_df"], 
                          membership = membership_val_ser, 
                          metric = "euc")
        # Invalid metric value
        self.assertRaises(ValueError, dp.centroid_distance, 
                          cents = cents_df[cents_df.columns[0:ntraits-2]], 
                          data = data["eff_df"], 
                          membership = membership_ser, 
                          metric = "invalid")
    def test_calc_medoids(self):
        """ Check the compute_pca creates a Dataframe"""
        # Create dummy data for testing
        data = ds.load_association_data(path_dir = "./data/TestData/",
                                                eff_fname = "unstdBeta_df.csv",
                                                exp_fname = "Beta_EXP.csv")
        npoints, _ = data["eff_df"].shape
        nclusts = 4
        membership = [rnd.randint(1, nclusts) for j in range(1, npoints + 1)]
        membership_ser = pd.Series(index = data["eff_df"].index,
                                     data = membership)
        dist_df= pd.DataFrame( data = ds.mat_dist(data["eff_df"]),
                              index = data["eff_df"].index,
                              columns = data["eff_df"].index)
        meds_df = dp.calc_medoids(data = data["eff_df"], data_dist = dist_df, membership = membership_ser)
        # Check return type is Dataframe
        self.assertTrue(isinstance(meds_df, pd.DataFrame))
        # Check the dimensions of the output
        self.assertTrue(meds_df.shape[0] == nclusts)
        self.assertTrue(meds_df.shape[1] == data["eff_df"].shape[1])
        # NEGATIVE TESTS
        # TypeErrors
        # data not dataframe
        self.assertRaises(TypeError, dp.calc_medoids, 
                          data = data["eff_df"].to_numpy(),
                          data_dist = dist_df,
                          membership = membership_ser)
        # dist_df not dataframe
        self.assertRaises(TypeError, dp.calc_medoids, 
                          data = data["eff_df"],
                          data_dist = dist_df.to_numpy(),
                          membership = membership_ser)
        # dist_df not dataframe
        self.assertRaises(TypeError, dp.calc_medoids, 
                          data = data["eff_df"],
                          data_dist = dist_df,
                          membership = membership_ser.to_numpy())
        # Value Errors
        # The dimension of the data against the membership dataframe does not match
        self.assertRaises(ValueError, dp.calc_medoids, 
                          data = data["eff_df"],
                          data_dist = dist_df,
                          membership = membership_ser.iloc[0:-2])
    def test_overlap_score(self):
        """The overlap score takes a dataframe of comparison values and evaluates an overall score.
        
        Check the input is a dataframe and output is a float. """
        data = ds.load_association_data(path_dir = "./data/TestData/",
                                                eff_fname = "unstdBeta_df.csv",
                                                exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        nclust1 = 6
        nclust2 = 4
        df1 = pd.DataFrame(index = data["eff_df"].index,
                           data = {"clust_num": [rnd.randint(1, nclust1) for i in range(ndata)],
                                   "clust_dist": [rnd.random() for j in range(ndata)]})
        df2 = pd.DataFrame(index = data["eff_df"].index,
                           data = {"clust_num": [rnd.randint(1, nclust2) for i in range(ndata)],
                                   "clust_dist": [rnd.random() for j in range(ndata)]})
        lab1 = "clust_num"
        lab2 = "clust_num"
        comp_df =pd.DataFrame(data = ds.compare_df1_to_df2(df1, df2, lab1, lab2))
        comp_val = dp.overlap_score(comp_df)
        # Check output is a float
        self.assertTrue(isinstance(comp_val, float))
        # Check for TypeError when input is not a dataframe
        self.assertRaises(TypeError, dp.overlap_score, comp_df.to_numpy())
    def test_overlap_pairs(self):
        """Overlap_pairs takes the input of percentage overlap between clusters for two different methods.
        Then returns the pairs which best match each other."""
        data = ds.load_association_data(path_dir = "./data/TestData/",
                                                eff_fname = "unstdBeta_df.csv",
                                                exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        nclust1 = 6
        nclust2 = 4
        df1 = pd.DataFrame(index = data["eff_df"].index,
                           data = {"clust_num": [rnd.randint(1, nclust1) for i in range(ndata)],
                                   "clust_dist": [rnd.random() for j in range(ndata)]})
        df2 = pd.DataFrame(index = data["eff_df"].index,
                           data = {"clust_num": [rnd.randint(1, nclust2) for i in range(ndata)],
                                   "clust_dist": [rnd.random() for j in range(ndata)]})
        lab1 = "clust_num"
        lab2 = "clust_num"
        comp_df =pd.DataFrame(data = ds.compare_df1_to_df2(df1, df2, lab1, lab2))
        comp_pairs = dp.overlap_pairs(comp_df, "method_A", "method_B")
        # Check output is a float
        self.assertTrue(isinstance(comp_pairs, pd.DataFrame))
        # Check output has the same number of rows as the in
        # put
        self.assertTrue(comp_pairs.shape[0]==comp_df.shape[0])
        # Check for TypeError when input is not a dataframe
        self.assertRaises(TypeError, dp.overlap_pairs, 
                          comp_percent_df = comp_df.to_numpy(),
                          meth_lab = "A",
                          meth_sec_lab = "B")
        # Check for TypeError when input label is not a string
        self.assertRaises(TypeError, dp.overlap_pairs, 
                          comp_percent_df = comp_df,
                          meth_lab = 1,
                          meth_sec_lab = "B")
        # Check for TypeError when input label is not a string
        self.assertRaises(TypeError, dp.overlap_pairs, 
                          comp_percent_df = comp_df,
                          meth_lab = "A",
                          meth_sec_lab = 1)
    def test_calc_per_from_comp(self):
        """The function `calc_per_from_comp should take the number 
        of points in an overlap between clusters and compute this 
        as a percentage of the number in the clusters overall 
        (the intersection divided by the union)."""
        data = ds.load_association_data(path_dir = "./data/TestData/",
                                                eff_fname = "unstdBeta_df.csv",
                                                exp_fname = "Beta_EXP.csv")
        ndata = data["eff_df"].shape[0]
        nclust1 = 6
        nclust2 = 4
        df1 = pd.DataFrame(index = data["eff_df"].index,
                           data = {"clust_num": [rnd.randint(1, nclust1) for i in range(ndata)],
                                   "clust_dist": [rnd.random() for j in range(ndata)]})
        df2 = pd.DataFrame(index = data["eff_df"].index,
                           data = {"clust_num": [rnd.randint(1, nclust2) for i in range(ndata)],
                                   "clust_dist": [rnd.random() for j in range(ndata)]})
        lab1 = "clust_num"
        lab2 = "clust_num"
        comp_df =pd.DataFrame(data = ds.compare_df1_to_df2(df1, df2, lab1, lab2))
        comp_per_df = dp.calc_per_from_comp(comp_df)
        # Check the output is a dataframe
        self.assertTrue(isinstance(comp_per_df, pd.DataFrame))
        # Check that a type error is raised if input is not a dataframe
        self.assertRaises(TypeError, dp.calc_per_from_comp, comp_df.to_numpy())
    def test_calc_medoids(self):
        """The function calc_medoids should take in the dataframe for the full data and the
        distances between points, and the cluster membership. It should locate the median datapoint
        for each cluster and output as a dataframe."""
        data = ds.load_association_data(path_dir = "./data/TestData/",
                                                eff_fname = "unstdBeta_df.csv",
                                                exp_fname = "Beta_EXP.csv")
        npoints, _ = data["eff_df"].shape
        nclusts = 4
        membership = [rnd.randint(1, nclusts) for j in range(1, npoints + 1)]
        membership_ser = pd.Series(index = data["eff_df"].index,
                                     data = membership)
        dist_df= pd.DataFrame( data = ds.mat_dist(data["eff_df"]),
                              index = data["eff_df"].index,
                              columns = data["eff_df"].index)
        meds = dp.calc_medoids(data=data["eff_df"], data_dist = dist_df, membership= membership_ser)
        # Check the output is a dataframe
        self.assertTrue(isinstance(meds, pd.DataFrame))
        # Check that all clusters have a corresponding medoid
        self.assertTrue(all([clust in meds.index for clust in membership_ser.unique()]))
        # Check that TypeError is raised if data is not a dataframe
        self.assertRaises(TypeError, dp.calc_medoids,
                          data = data["eff_df"].to_numpy(),
                          data_dist = dist_df,
                          membership = membership_ser)
        # Check that TypeError is raised if dist_df is not a dataframe
        self.assertRaises(TypeError, dp.calc_medoids,
                          data = data["eff_df"],
                          data_dist = dist_df.to_numpy(),
                          membership = membership_ser)
        # Check that TypeError is raised if membership is not a series
        self.assertRaises(TypeError, dp.calc_medoids,
                          data = data["eff_df"],
                          data_dist = dist_df,
                          membership = membership_ser.to_numpy())
        # Check that ValueError is raised if data and don't have matching number of rows
        self.assertRaises(ValueError, dp.calc_medoids,
                          data = data["eff_df"].iloc[0:-5,:],
                          data_dist = dist_df,
                          membership = membership_ser)
        # Check that ValueError is raised if the the number of rows and columns don't match in dist
        self.assertRaises(ValueError, dp.calc_medoids,
                          data = data["eff_df"],
                          data_dist = dist_df.iloc[0:-5,:],
                          membership = membership_ser)
        # Check that ValueError is raised if data and membership don't have matching number of rows.
        self.assertRaises(ValueError, dp.calc_medoids,
                          data = data["eff_df"],
                          data_dist = dist_df,
                          membership = membership_ser.iloc[0:-5])
if __name__ == '__main__':
    unittest.main()