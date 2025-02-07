"""
Author: Hayley Wragg
Created: 6th February 2025
Description:
    This module contains unit tests for the pathway enrichment functions in the 
    `pathway_enrichment` module. The tests cover various aspects of pathway enrichment 
    including fetching enrichment data, enriching clusters, applying p-value filters, 
    and removing pathway children.

Classes:
    TestPathwayEnrichment: A unittest.TestCase subclass that contains test methods for 
        pathway enrichment functions.

Test Methods:
    setUp: Initializes the test environment before each test method.
    tearDown: Cleans up the test environment after each test method.
    test_get_pathway_rows_from_data: Tests the `get_pathway_rows_from_data` function.
    test_fetch_enrichment: Tests the `fetch_enrichment` function.
    test_enrich_clust: Tests the `enrich_clust` function.
    test_enrich_method: Tests the `enrich_method` function.
    test_full_pathway_enrichment: Tests the `full_pathway_enrichment` function.
    test_apply_p_filter: Tests the `apply_p_filter` function.
    test_remove_paths_children: Tests the `remove_paths_children` function.
"""
import os
import shutil

import unittest
import random
import requests
import pandas as pd

from multitraitclustering import pathway_enrichment as pe

class TestPathwayEnrichment(unittest.TestCase):
    """
    Test suite for pathway enrichment functions.
    This class contains unit tests for various functions related to pathway enrichment.
    It sets up necessary data structures and directories for testing and cleans up after tests.
    Attributes:
        path_dir (str): Directory path for temporary test files.
        meth_key (str): Key for the method being tested.
        n_meths (int): Number of methods.
        n_clust (int): Number of clusters.
        n_paths (int): Number of pathways.
        n_pairs (int): Number of pairs.
        n_parents (int): Number of parent pathways.
        meth_labs (list): List of method labels.
        path_labs (list): List of pathway labels.
        path_dict (dict): Dictionary containing pathway data for each method.
        pathways_relation_df (pd.DataFrame): DataFrame containing parent-child pathway relationships
    Methods:
        setUp(): Sets up the test environment.
        tearDown(): Cleans up the test environment.
        test_get_pathway_rows_from_data(): Tests the get_pathway_rows_from_data function.
        test_fetch_enrichment(): Tests the fetch_enrichment function.
        test_enrich_clust(): Tests the enrich_clust function.
        test_enrich_method(): Tests the enrich_method function.
        test_full_pathway_enrichment(): Tests the full_pathway_enrichment function.
        test_apply_p_filter(): Tests the apply_p_filter function.
        test_remove_paths_children(): Tests the remove_paths_children function.
    """

    def setUp(self):
        self.path_dir = "./test_tmp"
        self.meth_key = "method_0"
        self.n_meths = 5
        self.n_clust = 4
        self.n_paths = 202
        self.n_pairs = 500
        self.n_parents = 30
        self.meth_labs = [f"method_{i}" for i in range(self.n_meths)]
        self.path_labs = [f"pathway_{j}" for j in range(self.n_paths)]
        self.path_dict = {meth:
            pd.DataFrame.from_dict({
                'path_id': [random.choice(self.path_labs) for _ in range(self.n_pairs)],
                'ClusterNumber': [random.choice(range(self.n_clust)) for _ in range(self.n_pairs)],
                'combined_score': [random.random() for _ in range(self.n_pairs)]}
                ) for meth in self.meth_labs
        }
        self.pathways_relation_df = pd.DataFrame(data= {
            'parent':[random.choice(self.path_labs[0:10]) for _ in range(self.n_parents)],
            'child': [random.choice(self.path_labs[10:-1]) for _ in range(self.n_parents)]
            }
        )
        if not os.path.exists(self.path_dir):
            os.makedirs(self.path_dir)

    def tearDown(self):
        if os.path.exists(self.path_dir):
            shutil.rmtree(self.path_dir)

    def test_get_pathway_rows_from_data(self):
        """
        Test the `get_pathway_rows_from_data` function from the `pe` module.
        This test checks the following:
        - The function returns a dictionary.
        - The dictionary contains the keys "or_row", "score_row", and "all_row".
        - raises a TypeError when the first argument is not a list.
        - raises a ValueError when the first argument is an empty list.
        - raises a ValueError if the following terms aren't in the first argument:
            'rank, pathway, pval, OR, score, overlap_genes, adjust_pval'
        Test Data:
        - data: [1, "pathway R-12345", 0.01, 2.5, 3.0, ["gene1", "gene2"], 0.05]
        - c_num_lab: 1
        """

        data = [1, "pathway R-12345", 0.01, 2.5, 3.0, ["gene1", "gene2"], 0.05]
        c_num_lab = 1
        result = pe.get_pathway_rows_from_data(data, c_num_lab)
        self.assertTrue(isinstance(result, dict))
        data_extra =  [1, "pathway R-12345", 0.01, 2.5, 3.0, ["gene1", "gene2"], 0.05, "extra", "more"]
        result_extra = pe.get_pathway_rows_from_data(data_extra, c_num_lab)
        self.assertTrue(isinstance(result_extra, dict))
        self.assertIn("or_row", result)
        self.assertIn("score_row", result)
        self.assertIn("all_row", result)
        # Negative checks
        self.assertRaises(TypeError, pe.get_pathway_rows_from_data, "not_a_list", c_num_lab)
        data_missing = [1, "pathway R-12345", 0.01, 2.5, 3.0 ]
        self.assertRaises(ValueError, pe.get_pathway_rows_from_data, data_missing, c_num_lab)
        self.assertRaises(ValueError, pe.get_pathway_rows_from_data, [], c_num_lab)

    def test_fetch_enrichment(self):
        """
        Test the fetch_enrichment function from the pe module.
        This test verifies the following:
        - The function returns a requests.Response object when called with valid parameters.
        - raises a TypeError when the session parameter is not a requests.Session object.
        - raises a ValueError when the gene_library parameter is an empty string.
        - raises a ValueError when the user_id parameter is an empty string.
        Raises:
            AssertionError: If any of the assertions fail.
        """
        req_ses = requests.Session()
        gene_library = "Reactome_2022"
        user_id = "12345"
        result = pe.fetch_enrichment(req_ses, gene_library, user_id)
        self.assertTrue(isinstance(result, requests.Response))
        # Negative checks
        self.assertRaises(TypeError, pe.fetch_enrichment, "not_a_session", gene_library, user_id)

    def test_enrich_clust(self):
        """
        Test the enrich_clust function from the pathway enrichment module.
        This test verifies the following:
        - The function returns a dictionary.
        - The dictionary contains the keys: "all_list", "OR_list", "score_list", and "user_list_id".
        - The function raises a TypeError when:
            - gene_set is not a list.
            - meth_key is not a valid type.
            - gene_library is not a valid type.
            - req_ses is not a requests.Session object.
        Raises:
            AssertionError: If any of the assertions fail.
            TypeError: If invalid arguments are passed to the enrich_clust function.
        """

        req_ses = requests.Session()
        gene_set = ["gene1", "gene2", "gene3"]
        c_num_lab = "1"
        gene_library = "Reactome_2022"
        result = pe.enrich_clust(gene_set, self.meth_key, c_num_lab, gene_library, req_ses)
        self.assertTrue(isinstance(result, dict))
        self.assertIn("all_list", result)
        self.assertIn("OR_list", result)
        self.assertIn("score_list", result)
        self.assertIn("user_list_id", result)
        # Negative checks
        self.assertRaises(TypeError, pe.enrich_clust,
                          gene_set = "not_a_list",
                          meth_key = self.meth_key,
                          c_num_lab = c_num_lab,
                          gene_library = gene_library,
                          req_ses = req_ses)
        # TypeError Invalid method ket
        self.assertRaises(TypeError, pe.enrich_clust,
                          gene_set = gene_set,
                          meth_key = 123,
                          c_num_lab = c_num_lab,
                          gene_library = gene_library,
                          req_ses = req_ses)
        # TypeError Invalid type for gene_library
        self.assertRaises(TypeError, pe.enrich_clust,
                          gene_set = gene_set,
                          meth_key = self.meth_key,
                          c_num_lab = c_num_lab,
                          gene_library = 123,
                          req_ses = req_ses)
        # TypeError req_ses not a session
        self.assertRaises(TypeError, pe.enrich_clust,
                          gene_set = gene_set,
                          meth_key = self.meth_key,
                          c_num_lab = c_num_lab,
                          gene_library = gene_library,
                          req_ses = "not_a_session")

    def test_enrich_method(self):
        """
        Test the enrich_method function from the pathway enrichment module.
        This test verifies the following:
        - The function returns a dictionary.
        - The dictionary contains the keys: "user_ids_list", "or_df", "score_df", and "all_df".
        - The function raises a TypeError when:
            - meth_key is not a string.
            - gene_clust_dict is not a dictionary.
            - gene_set_library is not a string.
            - req_ses is not a requests.Session object.
                Raises:
        AssertionError: If any of the assertions fail.
        TypeError: If invalid arguments are passed to the enrich_clust function.
        """
        req_ses = requests.Session()
        gene_clust_dict = {self.meth_key: pd.DataFrame({
            'Cluster1': [1, 0, 1],
            'Cluster2': [0, 1, 0]
        }, index=["gene1", "gene2", "gene3"])}
        gene_set_library = "Reactome_2022"
        result = pe.enrich_method(self.meth_key, gene_clust_dict, gene_set_library, req_ses)
        self.assertTrue(isinstance(result, dict))
        self.assertIn("user_ids_list", result)
        self.assertIn("or_df", result)
        self.assertIn("score_df", result)
        self.assertIn("all_df", result)
        # Negative checks
        self.assertRaises(TypeError, pe.enrich_method,
                  meth_key=123,
                  gene_clust_dict=gene_clust_dict,
                  gene_set_library=gene_set_library,
                  req_ses=req_ses)
        self.assertRaises(TypeError, pe.enrich_method,
                  meth_key=self.meth_key,
                  gene_clust_dict="not_a_dict",
                  gene_set_library=gene_set_library,
                  req_ses=req_ses)
        self.assertRaises(TypeError, pe.enrich_method,
                  meth_key=self.meth_key,
                  gene_clust_dict=gene_clust_dict,
                  gene_set_library=123,
                  req_ses=req_ses)
        self.assertRaises(TypeError, pe.enrich_method,
                  meth_key=self.meth_key,
                  gene_clust_dict=gene_clust_dict,
                  gene_set_library=gene_set_library,
                  req_ses="not_a_session")


    def test_full_pathway_enrichment(self):
        """Test the full_pathway_enrichment function.

            This test verifies that the full_pathway_enrichment function returns a dictionary
            containing the 'all_dict', 'or_dict', and 'score_dict' keys. It also checks
            that the returned value is a dictionary.
            """
        clust_dict = {self.meth_key: pd.DataFrame({
            'Cluster1': [1, 0, 1],
            'Cluster2': [0, 1, 0]
        }, index=["gene1", "gene2", "gene3"])}
        result = pe.full_pathway_enrichment(clust_dict, self.path_dir)
        self.assertTrue(isinstance(result, dict))
        self.assertIn("all_dict", result)
        self.assertIn("or_dict", result)
        self.assertIn("score_dict", result)

    def test_apply_p_filter(self):
        """Tests the apply_p_filter function.
            This function tests the `apply_p_filter` function with various scenarios,
            including:
            - Checking if the returned value is a dictionary with the correct keys.
            - Checking for TypeErrors when incorrect input types are provided.
            - Checking for ValueErrors when the input dictionaries do not contain the
              expected keys or columns.
            """

        all_dict = {self.meth_key: pd.DataFrame({
            'path_id': ["path1", "path2"],
            'pval': [1E-9, 1E-7]
        })}
        or_dict = {self.meth_key: pd.DataFrame({
            'path_id': ["path1", "path2"]
        })}
        score_dict = {self.meth_key: pd.DataFrame({
            'path_id': ["path1", "path2"]
        })}
        result = pe.apply_p_filter(self.meth_key, all_dict, or_dict, score_dict)
        self.assertTrue(isinstance(result, dict))
        self.assertIn("all_dict", result)
        self.assertIn("or_dict", result)
        self.assertIn("score_dict", result)
        # Negative checks
        self.assertRaises(KeyError, pe.apply_p_filter,
              meth_key = 123,
              all_dict = all_dict,
              or_dict = or_dict,
              score_dict = score_dict)
        self.assertRaises(TypeError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict="not_a_dict",
              or_dict = or_dict,
              score_dict = score_dict)
        self.assertRaises(TypeError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = all_dict,
              or_dict = "not_a_dict",
              score_dict = score_dict)
        self.assertRaises(TypeError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = all_dict,
              or_dict = or_dict,
              score_dict = "not_a_dict")
        # Check that the method key is in each dictionary
        invalid_dict = {"wrong_key": pd.DataFrame({
            'path_id': ["path1", "path2"],
            'pval': [1E-9, 1E-7]
        })}
        self.assertRaises(KeyError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict=invalid_dict,
              or_dict = or_dict,
              score_dict = score_dict)
        self.assertRaises(KeyError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = all_dict,
              or_dict = invalid_dict,
              score_dict = score_dict)
        self.assertRaises(KeyError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = all_dict,
              or_dict = or_dict,
              score_dict=invalid_dict)
        # Check that the method key is in each dictionary
        invalid_no_path_id_dict = {self.meth_key: pd.DataFrame({
            'not_path': ["path1", "path2"],
            'pval': [1E-9, 1E-7]
        })}
        # Check that the path_id column is in the all_dict
        self.assertRaises(ValueError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = invalid_no_path_id_dict,
              or_dict = or_dict,
              score_dict = score_dict)
        # Check that the path_id column is in the or_dict
        self.assertRaises(ValueError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = all_dict,
              or_dict = invalid_no_path_id_dict,
              score_dict = score_dict)
        # Check that the path_id column is in the score_dict
        self.assertRaises(ValueError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = all_dict,
              or_dict = or_dict,
              score_dict = invalid_no_path_id_dict)
            # Check that the method key is in each dictionary
        invalid_no_pval_dict = {self.meth_key: pd.DataFrame({
            'path_id': ["path1", "path2"],
            'not_pval': [1E-9, 1E-7]
        })}
        # Check that the pval column is in the all_dict
        self.assertRaises(ValueError, pe.apply_p_filter,
              meth_key=self.meth_key,
              all_dict = invalid_no_pval_dict,
              or_dict = or_dict,
              score_dict=all_dict)

    def test_remove_paths_children(self):
        """
        Test the `remove_paths_children` function from the `pe` module.
        This test verifies the following:
        - The function returns a dictionary.
        - raises a `TypeError` if `path_dir` is not a string.
        - raises a `TypeError` if `meth_key` is not a string.
        - raises a `TypeError` if `path_dict` is not a dictionary.
        - raises a `TypeError` if `pathways_relation_df` is not a DataFrame.
        - raises a `ValueError` if `meth_key` is not in `path_dict`.
        - raises a `ValueError` if `path_dict` is empty.
        - raises a `ValueError` if `pathways_relation_df` does not have the required columns.
        - raises a `ValueError` if DataFrames in `path_dict` do not have a 'path_id' column.
        """

        result = pe.remove_paths_children(self.path_dir,
                                          self.meth_key,
                                          self.path_dict,
                                          self.pathways_relation_df)
        # Check that out is a dict
        self.assertTrue(isinstance(result, dict))
        # ----------------
        # Negative checks
        # path_dir should be a string
        self.assertRaises(TypeError, pe.remove_paths_children,
                  path_dir=123,
                  meth_key=self.meth_key,
                  path_dict=self.path_dict,
                  pathways_relation_df=self.pathways_relation_df)

        # meth_key should be a string
        self.assertRaises(TypeError, pe.remove_paths_children,
                  path_dir=self.path_dir,
                  meth_key=123,
                  path_dict=self.path_dict,
                  pathways_relation_df=self.pathways_relation_df)

        # path_dict should be a dict
        self.assertRaises(TypeError, pe.remove_paths_children,
                  path_dir=self.path_dir,
                  meth_key=self.meth_key,
                  path_dict="not_a_dict",
                  pathways_relation_df=self.pathways_relation_df)

        # pathways_relation_df should be a DataFrame
        self.assertRaises(TypeError, pe.remove_paths_children,
                  path_dir=self.path_dir,
                  meth_key=self.meth_key,
                  path_dict=self.path_dict,
                  pathways_relation_df="not_a_dataframe")

        # meth_key should be in path_dict
        self.assertRaises(KeyError, pe.remove_paths_children,
                  path_dir=self.path_dir,
                  meth_key="non_existent_key",
                  path_dict=self.path_dict,
                  pathways_relation_df=self.pathways_relation_df)

        # path_dict should not be empty
        self.assertRaises(KeyError, pe.remove_paths_children,
                  path_dir=self.path_dir,
                  meth_key=self.meth_key,
                  path_dict={},
                  pathways_relation_df=self.pathways_relation_df)

        # pathways_relation_df should have required columns
        self.assertRaises(ValueError, pe.remove_paths_children,
                  path_dir=self.path_dir,
                  meth_key=self.meth_key,
                  path_dict=self.path_dict,
                  pathways_relation_df=pd.DataFrame())

        # DataFrames in path_dict should have 'path_id' column
        invalid_p_dict = self.path_dict.copy()
        invalid_p_dict[self.meth_key] = invalid_p_dict[self.meth_key].drop(columns=['path_id'])
        self.assertRaises(ValueError, pe.remove_paths_children,
              path_dir=self.path_dir,
              meth_key=self.meth_key,
              path_dict=invalid_p_dict,
              pathways_relation_df=self.pathways_relation_df)

if __name__ == '__main__':
    unittest.main()
