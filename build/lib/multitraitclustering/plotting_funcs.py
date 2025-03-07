"""
Author: Hayley Wragg
Created: 6th February 2025
Description:
    This module provides a collection of functions for visualizing clustering results using Altair.
    Includes functions for creating scatter plots of clusters, generating multiple scatter 
    plots with a fixed x-axis, comparing clustering methods using heatmaps, 
    and visualizing cluster pathways.
    Module relies on Altair library to create visualizations and Pandas for data manipulation.
"""

import altair as alt
import numpy as np
import pandas as pd


def chart_clusters(
    data,
    title,
    color_var,
    tooltip,
    col1="pc_1",
    col2="pc_2",
):
    """Plots the data as a scatter plot coloured by the cluster labels.

    The default is to use columns labeled pc_1 and pc_2 indicating the dominant principal
    components. However for smaller dimensions you may wish to use explicit axes. Set these with
    col1 = 'x_label', col2 = 'y_label'.

    Args:
        data: Data including the clusters and original source.
        title: plot title.
        color_var: label for the variable to group colours by.
        tooltip: list of labels for data to show when hovered.
        col1: column label for the x-axis, defaults to "pc_1".
        col2: column label for the y axis, defaults to "pc_2".

    Returns:
        An altair chart object.

    """
    # Input checks
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    if not isinstance(title, str):
        raise TypeError("Title must be a string.")
    if not isinstance(color_var, str):
        raise TypeError("color_var must be a string.")
    if not isinstance(tooltip, list):
        raise TypeError("Tooltip must be a list.")
    if not all(isinstance(item, str) for item in tooltip):
        raise TypeError("All items in tooltip must be strings.")
    if not isinstance(col1, str):
        raise TypeError("col1 must be a string")
    if not isinstance(col2, str):
        raise TypeError("col2 must be a string")
    if color_var not in data.columns:
        raise KeyError("color_var must be a column in data.")
    if col1 not in data.columns:
        raise KeyError("col1 must be a column in data.")
    if col2 not in data.columns:
        raise KeyError("col2 must be a column in data.")
    if not all(col in data.columns for col in tooltip):
        error_string = f"""All items in tooltip must be a column in data.
            Available columns: {data.columns}"""
        raise KeyError(error_string)
    data[color_var] = data[color_var].astype(str)
    chart = (
        alt.Chart(data, title=title)
        .mark_circle(size=60)
        .encode(x=col1, y=col2, color=color_var, tooltip=tooltip)
        .interactive()
    )
    return chart


def chart_clusters_multi(
    data,
    title,
    color_var,
    tooltip,
    xcol=None,
    col_list=None,
):
    """Generates multiple scatter plots with a fixed x-axis and varying y-axes, colored by
    cluster labels.

    This function iterates through a list of columns, creating a scatter plot for each column
    against a fixed x-axis. The plots are colored according to a specified variable,
    typically cluster assignments.

    Args:
        data (pd.DataFrame): Data including the clusters and original source.
        title (str): Plot title.
        color_var (str): Label for the variable to group colors by.
        tooltip (list): List of labels for data to show when hovered.
        xcol (str, optional): Label for fixed x column. Defaults to first column in the data.
        col_list (list, optional): List of columns for the y-axis. Defaults to [].

    Returns:
        dict: A dictionary where keys are column names from `col_list` and values are
        corresponding Altair chart objects.
    """
    # Input checks
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    if not isinstance(title, str):
        raise TypeError("Title must be a string.")
    if not isinstance(color_var, str):
        raise TypeError("color_var must be a string.")
    if not isinstance(tooltip, list):
        raise TypeError("Tooltip must be a list.")
    if not all(isinstance(item, str) for item in tooltip):
        raise TypeError("All items in tooltip must be strings.")
    if col_list is not None and not isinstance(col_list, list):
        raise TypeError("col_list must be a list")
    if not color_var in data.columns:
        error_string = f"""The columns {color_var} is not in the data.
            Available columns: {data.columns}"""
        raise KeyError(error_string)
    if xcol is not None and not xcol in data.columns:
        error_string = f"""The columns {xcol} is not in the data.
            Available columns: {data.columns}"""
        raise KeyError(error_string)
    if col_list is not None and not all(col in data.columns for col in col_list):
        error_string = f"""All items in col_list must be a column in data.
            Available columns: {data.columns}"""
        raise KeyError(error_string)
    if not all(col in data.columns for col in tooltip):
        error_string = f"""All items in tooltip must be a column in data.
            Available columns: {data.columns}"""
        raise KeyError(error_string)

    # Start of plotting code
    if xcol is None:
        col1 = data.columns[0]
    else:
        col1 = xcol
    if col_list is None:
        col_list = data.columns[1:]
    chart_dict = {}
    for col2 in col_list:
        chart_dict[col2] = (
            alt.Chart(data, title=title)
            .mark_circle(size=60)
            .encode(x=col1, y=col2, color=color_var + ":N", tooltip=tooltip)
        )
    return chart_dict


def chart_cluster_compare(
    data_array, xlabels, ylabels, x_lab, y_lab, z_lab, text_precision=".0f"
):
    """chart_cluster_compare Heatmap of the comparison percentage overlap of two clustering methods

    Args:
        data_array (numpy array): Comparison data - long form -
            x_lab (clust_no. for method 1), y_lab (clust_no for method 2),
            z_lab - no. in intersection/ no. in union
        xlabels (list of strings or array of numbers): Cluster labels for the columns
        ylabels (list of strings or array of numbers): Cluster labels for the rows
        x_lab (string): Label to get the x data from
        y_lab (string): Label to get the y data from
        z_lab (string): Label for the column containing the overlap percentage
        text_precision (str, optional): Precision for printing numeric variables. Defaults to ".0f".

    Returns:
        alt.Chart: An Altair chart object representing the heatmap.
    """
    # Input Checks
    if not isinstance(data_array, np.ndarray):
        raise TypeError("data_array must be a numpy array.")
    if not isinstance(xlabels, (list, np.ndarray)):
        raise TypeError("xlabels must be a list or array")
    if not isinstance(ylabels, (list, np.ndarray)):
        raise TypeError("ylabels must be a list or array")
    if not isinstance(x_lab, str):
        raise TypeError("x_lab must be a string.")
    if not isinstance(y_lab, str):
        raise TypeError("y_lab must be a string.")
    if not isinstance(z_lab, str):
        raise TypeError("z_lab must be a string.")
    if not isinstance(text_precision, str):
        raise TypeError("text_precision must be a string.")
    if len(ylabels) != data_array.shape[0]:
        error_string = f"""Length of ylabels ({len(ylabels)}) must match no. of rows in
            data_array ({data_array.shape[0]})."""
        raise ValueError(error_string)
    if len(xlabels) != data_array.shape[1]:
        error_string = f"""Length of xlabels ({len(xlabels)}) must match no. of cols in
            data_array ({data_array.shape[1]})."""
        raise ValueError(error_string)
    # Structure the plotting data
    # Convert this grid to columnar data expected by Altair
    ylen = data_array.shape[0]
    xlen = data_array.shape[1]
    x, y = np.meshgrid(range(0, xlen), range(0, ylen))
    x_clust = {i: cn for i, cn in enumerate(xlabels)}
    y_clust = {j: cn for j, cn in enumerate(ylabels)}
    z = data_array
    source = pd.DataFrame(
        dtype=str, data={x_lab: x.ravel(), y_lab: y.ravel(), z_lab: z.ravel()}
    )
    for i in range(0, xlen):
        source.loc[source[x_lab] == i, x_lab] = x_clust[i]
    for j in range(0, ylen):
        source.loc[source[y_lab] == j, y_lab] = y_clust[j]
    # Plotting
    chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            alt.X(x_lab + ":O").title(x_lab),
            alt.Y(y_lab + ":O").title(y_lab),
            alt.Color(z_lab + ":Q").title(z_lab),
        )
        .properties(width=400, height=400)
    )
    text = chart.mark_text(baseline="middle").encode(
        alt.Text(z_lab + ":Q", format=text_precision),
        color=alt.condition(
            alt.datum[z_lab] > 40, alt.value("white"), alt.value("black")
        ),
    )
    return chart + text


def chart_cluster_pathway(
    data_array, x_lab, y_lab, z_lab, title_str, text_precision=".0f"
):
    """Generates a heatmap-like chart using Altair to visualize a cluster pathway.
    The chart displays the relationship between three variables (x, y, and z) from the input array.
    It uses rectangles to represent the combinations of x and y, with the color of the rectangle
    indicating the value of z.  Text annotations are added to each rectangle to display the z value.
    Args:
        data_array: A pandas DataFrame or similar data structure that can be processed by Altair.
            It should contain columns corresponding to x_lab, y_lab, and z_lab.
        x_lab (str): The name of the column in `data_array` to be used for the x-axis.
        y_lab (str): The name of the column in `data_array` to be used for the y-axis.
        z_lab (str): The name of the column in `data_array` to be used for color and annotation.
        title_str (str): The title of the chart.
        text_precision (str, optional):  format string to control precision of the text annotation.
            Defaults to ".0f" (no decimal places).
     Returns:
        alt.Chart: An Altair chart object representing cluster pathway heatmap.  This can be further
            modified or displayed using Altair's API.
    """
    # Input Checks
    if not isinstance(data_array, pd.DataFrame):
        raise TypeError("data_array must be a pandas DataFrame.")
    if not isinstance(x_lab, str):
        raise TypeError("x_lab must be a string.")
    if not isinstance(y_lab, str):
        raise TypeError("y_lab must be a string.")
    if not isinstance(z_lab, str):
        raise TypeError("z_lab must be a string.")
    if not isinstance(title_str, str):
        raise TypeError("title_str must be a string.")
    if not isinstance(text_precision, str):
        raise TypeError("text_precision must be a string.")
    if x_lab not in data_array.columns:
        raise KeyError("x_lab must be a column in data_array.")
    if y_lab not in data_array.columns:
        raise KeyError("y_lab must be a column in data_array.")
    if z_lab not in data_array.columns:
        raise KeyError("z_lab must be a column in data_array.")
    # Convert this grid to columnar data expected by Altair
    title = alt.TitleParams(title_str, anchor="middle")
    chart = (
        alt.Chart(data_array, title=title)
        .mark_rect()
        .encode(
            alt.X(x_lab + ":O").title(x_lab),
            alt.Y(y_lab + ":O").title(y_lab),
            alt.Color(z_lab + ":Q").title(z_lab),
        )
        .properties(width=400, height=400)
    )
    text = chart.mark_text(baseline="middle").encode(
        alt.Text(z_lab + ":Q", format=text_precision), color=alt.value("black")
    )
    return chart + text


def pathway_bars(df, xlab, ylab, group_lab, max_val, title):
    """Generates a bar chart visualizing pathway data.
    Args:
        df (pd.DataFrame): DataFrame containing the data for the chart.
            Must contain columns corresponding to xlab, ylab, and group_lab.
        xlab (str): Name of the column to use for the x-axis (pathway names).
        ylab (str): Name of the column to use for the y-axis (pathway values).
        group_lab (str): Name of the column to use for grouping the bars into
            separate subplots.
        max_val (float): Max value, y-vals greater than or equal to this value
            will be colored green.
        title (str): Title of the chart.
    Returns:
        alt.Chart: An Altair bar chart object. The chart is interactive, allowing
            for zooming and panning.
    """
    # Input checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(xlab, str):
        raise TypeError("xlab must be a string.")
    if not isinstance(ylab, str):
        raise TypeError("ylab must be a string.")
    if not isinstance(group_lab, str):
        raise TypeError("group_lab must be a string.")
    if not isinstance(max_val, (int, float)):
        raise TypeError("max_val must be a number.")
    if not isinstance(title, str):
        raise TypeError("title must be a string.")
    if xlab not in df.columns:
        raise KeyError("xlab must be a column in df.")
    if ylab not in df.columns:
        raise KeyError("ylab must be a column in df.")
    if group_lab not in df.columns:
        raise KeyError("group_lab must be a column in df.")
    chart = (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            x=alt.X(xlab + ":N", axis=alt.Axis(labels=False), title=None),
            y=ylab + ":Q",
            color=alt.condition(
                alt.datum[ylab]
                >= max_val,  # If the rating is less than the min it returns True,
                alt.value("green"),  # and the matching bars are set as green.
                # and if it does not satisfy the condition
                # the color is set to steelblue.
                alt.value("steelblue"),
            ),
            column=alt.Column(
                group_lab + ":N",
                header=alt.Header(
                    labelAngle=-90, orient="top", labelOrient="top", labelAlign="right"
                ),
                title=None,
            ),
        )
        .interactive()
    )
    return chart


def overlap_bars(df, xlab, ylab, group_lab, sep_lab, title):
    """Generates an Altair bar chart with overlapping bars, faceted by group.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to plot.
            xlab (str): col in DataFrame for the x-axis, categorical data.
            ylab (str): col in DataFrame for the y-axis, quantitative data.
            group_lab (str): col in DataFrame for chart groups, categorical data.
            sep_lab (str): col in DataFrame to separate bars within groups, categorical data.
            title (str): The title of the chart.

        Returns:
            alt.Chart: An Altair chart object representing the overlapping bar chart.
            The x-axis labels are hidden.
            The column headers are rotated and aligned for better readability.
    """
    # Input checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(xlab, str):
        raise TypeError("xlab must be a string.")
    if not isinstance(ylab, str):
        raise TypeError("ylab must be a string.")
    if not isinstance(group_lab, str):
        raise TypeError("group_lab must be a string.")
    if not isinstance(sep_lab, str):
        raise TypeError("sep_lab must be a string.")
    if not isinstance(title, str):
        raise TypeError("title must be a string.")
    if xlab not in df.columns:
        raise KeyError("xlab must be a column in df.")
    if ylab not in df.columns:
        raise KeyError("ylab must be a column in df.")
    if group_lab not in df.columns:
        raise KeyError("group_lab must be a column in df.")
    if sep_lab not in df.columns:
        raise KeyError("sep_lab must be a column in df.")
    chart = (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            x=alt.X(xlab + ":N", axis=alt.Axis(labels=False), title=None),
            y=ylab + ":Q",
            color=alt.Color(sep_lab + ":N", title=None),
            column=alt.Column(
                group_lab + ":N",
                header=alt.Header(
                    labelAngle=-90, orient="top", labelOrient="top", labelAlign="right"
                ),
                title=None,
            ),
        )
    )
    return chart
