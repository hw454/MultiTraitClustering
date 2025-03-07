"""
Author: Hayley Wragg
Created: 6th February 2025
Description:
    This module contains unit tests for the plotting functions in the MultiTraitClustering 
    package.
    The tests verify the functionality of the following functions:
    - `chart_clusters`: Tests the creation of scatter plots for cluster visualization.
    - `chart_clusters_multi`: Tests the creation of multiple scatter plots for cluster 
    visualization across different columns.
    - `chart_cluster_compare`: Tests the creation of heatmap-like charts for comparing clusters.
    - `chart_cluster_pathway`: Tests the creation of pathway analysis charts.
    The tests cover various scenarios, including valid parameters, custom palettes, and 
    negative test cases with invalid inputs.
"""

import unittest
import pandas as pd
import altair as alt
import numpy as np

from multitraitclustering import plotting_funcs as pc

class TestPlottingFuncs(unittest.TestCase):
    """
    Unit tests for the plotting functions in the MultiTraitClustering package.
    This class contains test methods to verify the functionality of the `chart_clusters` function,
    including its ability to generate Altair charts, handle different parameters, and raise
    appropriate exceptions for invalid inputs.
    """

    def test_chart_clusters(self):
        """
        Test the chart_clusters function.
        This function tests the chart_clusters function with various scenarios,
        including valid parameters, a custom palette, and negative test cases
        with invalid inputs.
        """

        # Create a sample DataFrame
        data = pd.DataFrame({
            'pc_1': [1, 2, 3, 4, 5],
            'pc_2': [5, 4, 3, 2, 1],
            'cluster': ['A', 'B', 'A', 'B', 'A'],
            'tooltip': ['info1', 'info2', 'info3', 'info4', 'info5']
        })

        # Call chart_clusters with valid parameters
        chart = pc.chart_clusters(data, 'Test Chart', 'cluster', ['tooltip'])

        # Assert the type of the returned object
        self.assertTrue(isinstance(chart, alt.Chart))

        # Negative test cases
        with self.assertRaises(TypeError):
            pc.chart_clusters(data.to_numpy(), 'Test Chart', 'cluster', ['tooltip'])

        with self.assertRaises(KeyError):
            pc.chart_clusters(data, 'Test Chart', 'invalid_column', ['tooltip'])

        with self.assertRaises(KeyError):
            pc.chart_clusters(data, 'Test Chart', 'cluster', ['invalid_tooltip'])

        with self.assertRaises(KeyError):
            pc.chart_clusters(data, 'Test Chart', 'cluster', ['tooltip'], col1 = 'invalid_column')

        with self.assertRaises(KeyError):
            pc.chart_clusters(data, 'Test Chart', 'cluster', ['tooltip'], col2 = 'invalid_column')

    def test_chart_clusters_multi(self):
        """Test the chart_clusters_multi function.
            This function tests the chart_clusters_multi function with different
            parameters to ensure it returns the correct type of object and that
            the charts are created correctly.
            """

        # Create a sample DataFrame
        data = pd.DataFrame({
            'pc_1': [1, 2, 3, 4, 5],
            'pc_2': [5, 4, 3, 2, 1],
            'pc_3': [2, 3, 1, 5, 4],
            'cluster': ['A', 'B', 'A', 'B', 'A'],
            'tooltip': ['info1', 'info2', 'info3', 'info4', 'info5']
        })

        # Call chart_clusters_multi with valid parameters
        charts = pc.chart_clusters_multi(data, 'Test Chart', 'cluster', ['tooltip'],
                                         col_list=['pc_2', 'pc_3'])

        # Assert the type of the returned object
        self.assertTrue(isinstance(charts, dict))
        self.assertEqual(len(charts), 2)
        self.assertTrue(all(isinstance(chart, alt.Chart) for chart in charts.values()))

        # Negative test cases
        with self.assertRaises(TypeError):
            pc.chart_clusters_multi(data.to_numpy(), 'Test Chart', 'cluster', ['tooltip'],
                         col_list=['pc_2', 'pc_3'])

        with self.assertRaises(KeyError):
            pc.chart_clusters_multi(data, 'Test Chart', 'invalid_column', ['tooltip'],
                         col_list=['pc_2', 'pc_3'])

        with self.assertRaises(KeyError):
            pc.chart_clusters_multi(data, 'Test Chart', 'cluster', ['invalid_tooltip'],
                         col_list=['pc_2', 'pc_3'])

        with self.assertRaises(KeyError):
            pc.chart_clusters_multi(data, 'Test Chart', 'cluster', ['tooltip'],
                         col_list=['invalid_column'])
    def test_chart_cluster_compare(self):
        """Test the chart_cluster_compare function.
            This test case creates a sample data array and calls the chart_cluster_compare
            function with valid parameters. It then asserts that the returned object
            is an instance of alt.LayerChart.
            """

        # Create a sample data array
        data_array = np.array([[0.1, 0.2, 0.3],[None, 2, 4], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        xlabels = ['X1', 'X2', 'X3']
        ylabels = ['Y1', 'Y2', 'Y3', ' Y4']
        x_lab = 'x_cluster'
        y_lab = 'y_cluster'
        z_lab = 'overlap'

        # Call chart_cluster_compare with valid parameters
        chart = pc.chart_cluster_compare(data_array, xlabels, ylabels, x_lab, y_lab, z_lab)

        # Assert the type of the returned object
        self.assertTrue(isinstance(chart, alt.LayerChart))

        # Negative test cases
        with self.assertRaises(TypeError):
            pc.chart_cluster_compare(data_array.tolist(), xlabels, ylabels, x_lab, y_lab, z_lab)

        with self.assertRaises(TypeError):
            pc.chart_cluster_compare(data_array, 1, ylabels, x_lab, y_lab, z_lab)

        with self.assertRaises(TypeError):
            pc.chart_cluster_compare(data_array, xlabels, 2, x_lab, y_lab, z_lab)

        with self.assertRaises(TypeError):
            pc.chart_cluster_compare(data_array, xlabels, ylabels, 3, y_lab, z_lab)

        with self.assertRaises(TypeError):
            pc.chart_cluster_compare(data_array, xlabels, ylabels, x_lab, 4, z_lab)

        with self.assertRaises(TypeError):
            pc.chart_cluster_compare(data_array, xlabels, ylabels, x_lab, y_lab, 5)

    def test_chart_cluster_pathway(self):
        """Test the chart_cluster_pathway function.
            This test case creates a sample DataFrame and calls the chart_cluster_pathway
            function with valid parameters. It then asserts that the returned object
            is an instance of alt.LayerChart.
            """

        # Create a sample DataFrame
        data = pd.DataFrame({
            'pathway1': ['A', 'B', 'C', 'A', 'B'],
            'pathway2': ['X', 'Y', 'Z', 'X', 'Y'],
            'value': [10, 20, 30, 40, 50]
        })
        x_lab = 'pathway1'
        y_lab = 'pathway2'
        z_lab = 'value'
        title_str = 'Pathway Analysis'

        # Call chart_cluster_pathway with valid parameters
        chart = pc.chart_cluster_pathway(data, x_lab, y_lab, z_lab, title_str)

        # Assert the type of the returned object
        self.assertTrue(isinstance(chart, alt.LayerChart))

        # Negative test cases
        with self.assertRaises(TypeError):
            pc.chart_cluster_pathway(data.to_numpy(), x_lab, y_lab, z_lab, title_str)

        with self.assertRaises(KeyError):
            pc.chart_cluster_pathway(data, 'invalid_column', y_lab, z_lab, title_str)

        with self.assertRaises(KeyError):
            pc.chart_cluster_pathway(data, x_lab, 'invalid_column', z_lab, title_str)

        with self.assertRaises(KeyError):
            pc.chart_cluster_pathway(data, x_lab, y_lab, 'invalid_column', title_str)

        with self.assertRaises(TypeError):
            pc.chart_cluster_pathway(data, x_lab, y_lab, z_lab, 123)
    def test_pathway_bars(self):
        """Test the pathway_bars function.
            This test case creates a sample DataFrame and calls the pathway_bars
            function with valid parameters. It then asserts that the returned object
            is an instance of alt.Chart.
            """

        # Create a sample DataFrame
        data = pd.DataFrame({
            'pathway': ['A', 'B', 'C', 'D', 'E'],
            'value': [10, 20, 30, 40, 50],
            'group': ['X', 'X', 'Y', 'Y', 'X']
        })
        xlab = 'pathway'
        ylab = 'value'
        group_lab = 'group'
        max_val = 30
        title = 'Pathway Analysis'

        # Call pathway_bars with valid parameters
        chart = pc.pathway_bars(data, xlab, ylab, group_lab, max_val, title)

        # Assert the type of the returned object
        self.assertTrue(isinstance(chart, alt.Chart))

        # Negative test cases
        with self.assertRaises(TypeError):
            pc.pathway_bars(data.to_numpy(), xlab, ylab, group_lab, max_val, title)

        with self.assertRaises(TypeError):
            pc.pathway_bars(data, 123, ylab, group_lab, max_val, title)

        with self.assertRaises(TypeError):
            pc.pathway_bars(data, xlab, 456, group_lab, max_val, title)

        with self.assertRaises(TypeError):
            pc.pathway_bars(data, xlab, ylab, 789, max_val, title)

        with self.assertRaises(TypeError):
            pc.pathway_bars(data, xlab, ylab, group_lab, 'abc', title)

        with self.assertRaises(TypeError):
            pc.pathway_bars(data, xlab, ylab, group_lab, max_val, 321)

        with self.assertRaises(KeyError):
            pc.pathway_bars(data, 'invalid_column', ylab, group_lab, max_val, title)

        with self.assertRaises(KeyError):
            pc.pathway_bars(data, xlab, 'invalid_column', group_lab, max_val, title)

        with self.assertRaises(KeyError):
            pc.pathway_bars(data, xlab, ylab, 'invalid_column', max_val, title)

    def test_overlap_bars(self):
        """Test the overlap_bars function.
            This test case creates a sample DataFrame and calls the overlap_bars
            function with valid parameters. It then asserts that the returned object
            is an instance of alt.Chart.
            """

        # Create a sample DataFrame
        data = pd.DataFrame({
                    'category': ['A', 'B', 'C', 'A', 'B'],
                    'value': [10, 20, 30, 40, 50],
                    'group': ['X', 'X', 'Y', 'Y', 'X'],
                    'separator': ['P', 'Q', 'P', 'Q', 'P']
        })
        xlab = 'category'
        ylab = 'value'
        group_lab = 'group'
        sep_lab = 'separator'
        title = 'Overlap Analysis'

        # Call overlap_bars with valid parameters
        chart = pc.overlap_bars(data, xlab, ylab, group_lab, sep_lab, title)

        # Assert the type of the returned object
        self.assertTrue(isinstance(chart, alt.Chart))

        # Negative test cases
        with self.assertRaises(TypeError):
            pc.overlap_bars(data.to_numpy(), xlab, ylab, group_lab, sep_lab, title)

        with self.assertRaises(TypeError):
            pc.overlap_bars(data, 123, ylab, group_lab, sep_lab, title)

        with self.assertRaises(TypeError):
            pc.overlap_bars(data, xlab, 456, group_lab, sep_lab, title)

        with self.assertRaises(TypeError):
            pc.overlap_bars(data, xlab, ylab, 789, sep_lab, title)

        with self.assertRaises(TypeError):
            pc.overlap_bars(data, xlab, ylab, group_lab, 910, title)

        with self.assertRaises(TypeError):
            pc.overlap_bars(data, xlab, ylab, group_lab, sep_lab, 321)

        with self.assertRaises(KeyError):
            pc.overlap_bars(data, 'invalid_column', ylab, group_lab, sep_lab, title)

        with self.assertRaises(KeyError):
            pc.overlap_bars(data, xlab, 'invalid_column', group_lab, sep_lab, title)

        with self.assertRaises(KeyError):
            pc.overlap_bars(data, xlab, ylab, 'invalid_column', sep_lab, title)

        with self.assertRaises(KeyError):
            pc.overlap_bars(data, xlab, ylab, group_lab, 'invalid_column', title)

if __name__ == '__main__':
    unittest.main()
