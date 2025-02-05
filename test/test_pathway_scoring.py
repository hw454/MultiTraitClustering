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
        pathways = ["pathway_%d"%i for i in range(npoints)]
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
       
if __name__ == '__main__':
    unittest.main()
