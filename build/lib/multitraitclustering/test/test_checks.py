import unittest
import pandas as pd

import checks as ch

class TestMultiTraitClustering(unittest.TestCase):

    def test_int_check(self):
        """The function int_check should raise a TypeError if not a integer is entered """
        # Check the function pass for an integer
        self.assertIsNone(ch.int_check(2, var_name = "var_name"))
        # Check that a TypeError is raised for not an integer
        self.assertRaises(TypeError, ch.int_check, 2.3, var_name = "var_name")
    def test_float_check(self):
        """The function float_check should raise a TypeError if not a float is entered"""
        # Check the function pass for an integer
        self.assertIsNone(ch.float_check(2.3, var_name = "var_name"))
        # Check that a TypeError is raised for not an float
        self.assertRaises(TypeError, ch.float_check, "A", var_name = "var_name")
    def test_str_check(self):
        """The function str_check should raise a TypeError if not a string is entered"""
        # Check the function pass for an integer
        self.assertIsNone(ch.str_check("A", var_name = "var_name"))
        # Check that a TypeError is raised for not an float
        self.assertRaises(TypeError, ch.str_check, 2, var_name = "var_name")
    def test_df_check(self):
        """The function df_check should raise a TypeError if not a Dataframe is entered"""
        # Check the function pass for an integer
        self.assertIsNone(ch.df_check(pd.DataFrame(), var_name = "var_name"))
        # Check that a TypeError is raised for not an Dataframe
        self.assertRaises(TypeError, ch.df_check, 2, var_name = "var_name")