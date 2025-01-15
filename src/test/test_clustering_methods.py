import unittest
import pandas as pd
import random as rnd
import numpy as np

import data_manipulation.data_setup as ds
import data_manipulation.data_processing as dp
import clustering.multi_trait_clustering as mtc
import clustering.clustering_methods as meth

class TestClusteringMethods(unittest.TestCase):
    def test_kmeans(self):
        """ Kmeans should take in the exposure data, distance data and results data frame.
        Then compute clusters using Kmeans and add them to the results dataframe.
        Returns a dictionary containing the results dataframe and the method parameters."""
        data = ds.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        dist_df= pd.DataFrame( data = ds.mat_dist(data["eff_df"]),
                              index = data["eff_df"].index,
                              columns = data["eff_df"].index)
        res_df = data["exp_df"].merge(data["eff_df"], left_index= True, right_index = True,
                   how='inner')
        kmeans_out = meth.kmeans(data["eff_df"], dist_df, res_df)
        # Check that a dictionary is output
        self.assertTrue(isinstance(kmeans_out, dict))
        # Check that the dictionary contains results dataframe
        self.assertTrue(isinstance(kmeans_out["results"], pd.DataFrame))
        # Check that the dictionary contains the cluster parameter dictionary
        self.assertTrue(isinstance(kmeans_out["cluster_dict"], dict))
        # TYPE CHECKS
        # Check that TypeError is returned if exposure data is not entered as a dataframe
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"].to_numpy(),
                          dist_df = dist_df, 
                          res_df = res_df)
        # Check that TypeError is returned if distance data is not entered as a dataframe
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df.to_numpy(), 
                          res_df = res_df)
        # Check that TypeError is returned if results data is not entered as a dataframe
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df.to_numpy())
        # Check that TypeError is returned if a non integer number of clusters is entered
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          nclust = 3.2)
        # Check that TypeError is returned if rand_st is not an integer
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          rand_st = 3.2)
        # Check that TypeError is returned if n_in is not an integer or string
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          n_in = 3.2)
        # Check that TypeError is returned if init_km is not a string or an array
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          init_km = 3.2)
        # Check that TypeError is returned if iter_max is not an integer
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          iter_max = 3.2)
        # Check that TypeError is returned if kmeans_alg is not a string
        self.assertRaises(TypeError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          kmeans_alg = 3.2)
        # VALUE CHECKS
        # Check that ValueError is returned if n_in is a string but not "auto".
        self.assertRaises(ValueError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          n_in = "A")
        # Check that ValueError is returned if init_km is not one of: ‘k-means++’, ‘random’ or an array
        self.assertRaises(ValueError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          init_km = "A")
        # Check that ValueError is returned if kmeans_alg is not one of: “lloyd”, “elkan”
        self.assertRaises(ValueError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          kmeans_alg = "A")
        # Check that ValueError is returned if dist_met is not "Euclidean" or "CosineSimilarity"
        self.assertRaises(ValueError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          dist_met = "A")
        # DIMENSION CHECKS
        # Incorrect rows in data
        self.assertRaises(ValueError, meth.kmeans, 
                          assoc_df = data["eff_df"].iloc[0:-2,:],
                          dist_df = dist_df, 
                          res_df = res_df)
        # Mismatch rows and columns in dist_df
        self.assertRaises(ValueError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df.iloc[0:-2,:], 
                          res_df = res_df)
        # Incorrect rows in result
        self.assertRaises(ValueError, meth.kmeans, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df.iloc[0:-2,:])
    def test_kmedoids(self):
        """ Kmedoids should take in the exposure data, distance data and results data frame.
        Then compute clusters using Kmedoids and add them to the results dataframe.
        Returns a dictionary containing the results dataframe and the method parameters."""
        data = ds.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        
        dist_df= pd.DataFrame( data = ds.mat_dist(data["eff_df"]),
                              index = data["eff_df"].index,
                              columns = data["eff_df"].index)
        res_df = data["exp_df"].merge(data["eff_df"], left_index= True, right_index = True,
                   how='inner')
        kmedoids_out = meth.kmedoids(data["eff_df"], dist_df, res_df)
        # Check that a dictionary is output
        self.assertTrue(isinstance(kmedoids_out, dict))
        # Check that the dictionary contains results dataframe
        self.assertTrue(isinstance(kmedoids_out["results"], pd.DataFrame))
        # Check that the dictionary contains the cluster parameter dictionary
        self.assertTrue(isinstance(kmedoids_out["cluster_dict"], dict))
        # TYPE CHECKS
        # Check that TypeError is returned if exposure data is not entered as a dataframe
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc_df = data["eff_df"].to_numpy(),
                          dist_df = dist_df, 
                          res_df = res_df)
        # Check that TypeError is returned if distance data is not entered as a dataframe
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df.to_numpy(), 
                          res_df = res_df)
        # Check that TypeError is returned if results data is not entered as a dataframe
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df.to_numpy())
        # Check that TypeError is returned if a non integer number of clusters is entered
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          nclust = 3.2)
        # Check that TypeError is returned if rand_st is not an integer
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          rand_st = 3.2)
        # Check that TypeError is returned if init_km is not a string or an array
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          init_kmed = 3.2)
        # Check that TypeError is returned if iter_max is not an integer
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          iter_max = 3.2)
        # Check that TypeError is returned if kmedoids_alg is not a string
        self.assertRaises(TypeError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          kmedoids_alg = 3.2)
        # VALUE CHECKS
        # Check that ValueError is returned if init_km is not one of: ‘k-means++’, ‘random’ or an array
        self.assertRaises(ValueError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          init_kmed = "A")
        # Check that ValueError is returned if kmeans_alg is not one of: “lloyd”, “elkan”
        self.assertRaises(ValueError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          kmedoids_alg = "A")
        # Check that ValueError is returned if dist_met is not "Euclidean" or "CosineSimilarity"
        self.assertRaises(ValueError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          dist_met = "A")
        # DIMENSION CHECKS
        # Incorrect rows in data
        self.assertRaises(ValueError, meth.kmedoids, 
                          assoc_df = data["eff_df"].iloc[0:-2,:],
                          dist_df = dist_df, 
                          res_df = res_df)
        # Mismatch rows and columns in dist_df
        self.assertRaises(ValueError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df.iloc[0:-2,:], 
                          res_df = res_df)
        # Incorrect rows in result
        self.assertRaises(ValueError, meth.kmedoids, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df.iloc[0:-2,:])
    def test_dbscan(self):
        """ DBSCAN should take in the exposure data, distance data and results data frame.
        Then compute clusters using DBSCAN and add them to the results dataframe.
        Returns a dictionary containing the results dataframe and the method parameters."""
        data = ds.load_association_data(path_dir = "../data/TestData/",
                                        eff_fname = "unstdBeta_df.csv",
                                        exp_fname = "Beta_EXP.csv")
        
        dist_df= pd.DataFrame( data = ds.mat_dist(data["eff_df"]),
                              index = data["eff_df"].index,
                              columns = data["eff_df"].index)
        res_df = data["exp_df"].merge(data["eff_df"], left_index= True, right_index = True,
                   how='inner')
        dbscan_out = meth.dbscan(data["eff_df"], dist_df, res_df)
        # Check that a dictionary is output
        self.assertTrue(isinstance(dbscan_out, dict))
        # Check that the dictionary contains results dataframe
        self.assertTrue(isinstance(dbscan_out["results"], pd.DataFrame))
        # Check that the dictionary contains the cluster parameter dictionary
        self.assertTrue(isinstance(dbscan_out["cluster_dict"], dict))
        # TYPE CHECKS
        # Check that TypeError is returned if exposure data is not entered as a dataframe
        self.assertRaises(TypeError, meth.dbscan, 
                          assoc_df = data["eff_df"].to_numpy(),
                          dist_df = dist_df, 
                          res_df = res_df)
        # Check that TypeError is returned if distance data is not entered as a dataframe
        self.assertRaises(TypeError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df.to_numpy(), 
                          res_df = res_df)
        # Check that TypeError is returned if results data is not entered as a dataframe
        self.assertRaises(TypeError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df.to_numpy())
        # Check that TypeError is returned if a non integer number of for min_s
        self.assertRaises(TypeError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          min_s = 3.2)
        # Check that TypeError is returned if eps is not a float
        self.assertRaises(TypeError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          eps = "A")
        # Check that TypeError is returned if db_alg is not a string
        self.assertRaises(TypeError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          db_alg = 3.2)
        # Check that TypeError is returned if iter_max is not an string
        self.assertRaises(TypeError, meth.dbscan, 
                          assoc = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          dist_met = 3.2)
        # VALUE CHECKS
        # Check that ValueError is returned if init_km is not one of: `auto`, `ball_tree`, `kd_tree`, `brute`
        self.assertRaises(ValueError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          db_alg = "A")
        # Check that ValueError is returned if dist_met is not "Euclidean" or "CosineSimilarity"
        self.assertRaises(ValueError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df,
                          dist_met = "A")
        # DIMENSION CHECKS
        # Incorrect rows in data
        self.assertRaises(ValueError, meth.dbscan, 
                          assoc_df = data["eff_df"].iloc[0:-2,:],
                          dist_df = dist_df, 
                          res_df = res_df)
        # Mismatch rows and columns in dist_df
        self.assertRaises(ValueError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df.iloc[0:-2,:], 
                          res_df = res_df)
        # Incorrect rows in result
        self.assertRaises(ValueError, meth.dbscan, 
                          assoc_df = data["eff_df"],
                          dist_df = dist_df, 
                          res_df = res_df.iloc[0:-2,:])

if __name__ == '__main__':
    unittest.main()