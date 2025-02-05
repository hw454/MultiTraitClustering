import altair as alt
import numpy as np
import pandas as pd

# TODO #7 Tests for chart_clusters
def chart_clusters(data, title, color_var, tooltip,
                   palette = None, clust_groups = None,
                   col1= "pc_1", col2= "pc_2"):
    """
    chart_clusters plots the data as a scatter plot coloured by the cluster labels

    _extended_summary_

    :param data: Data including the clusters and original source
    :type data: pd.Dataframe
    :param title: plot title
    :type title: string
    :param color_var: label for the variable to group colours by
    :type color_var: string
    :param tooltip: list of labels for data to show when hovered
    :type tooltip: list
    :param palette: colour palette, defaults to None
    :type palette: string, optional
    :param clust_groups: _description_, defaults to None
    :type clust_groups: string, optional
    :param col1: column label for the x-axis, defaults to "pc_1"
    :type col1: str, optional
    :param col2: column label for the y axis, defaults to "pc_2"
    :type col2: str, optional
    """
    if palette is not None:
        chart = alt.Chart(data, title=title).mark_circle(size=60).encode(
            x = col1,
            y = col2,
            color = alt.Color(color_var, scale=alt.Scale(domain=clust_groups, range=palette)),
            tooltip = tooltip
        ).interactive()
    else:
        chart = alt.Chart(data, title=title).mark_circle(size=60).encode(
            x = col1,
            y = col2,
            color = color_var,
            tooltip = tooltip
        ).interactive()

    return(chart)

# TODO #8 tests for chart_clusters_multi
def chart_clusters_multi(data, title, color_var, tooltip, xcol = None,
                        palette = None, clust_groups = None, col_list = []):
    """
    chart_clusters_multi Iterates through axis scatter plot for exposure + traits


    :param data: Data including the clusters and original source
    :type data: pd.Dataframe
    :param title: plot title
    :type title: string
    :param color_var: label for the variable to group colours by
    :type color_var: string
    :param tooltip: list of labels for data to show when hovered
    :type tooltip: list
    :param xcol: label for fixed x column, defaults to None
    :type xcol: string, optional
    :param palette: _description_, defaults to None
    :type palette: _type_, optional
    :param clust_groups: _description_, defaults to None
    :type clust_groups: _type_, optional
    :param col_list: list of columns for y column, defaults to []
    :type col_list: list, optional
    """
    if xcol is None:
        col1 = data.columns[0]
    else: col1 = xcol
    chart_dict = {}
    for col2 in col_list:
        if palette is not None:
            chart_dict[col2] = alt.Chart(data, title=title).mark_circle(size=60).encode(
                x = col1,
                y = col2,
                color = alt.Color(color_var, scale=alt.Scale(domain=clust_groups, range=palette)),
                tooltip = tooltip
            )
        else:
            chart_dict[col2] = alt.Chart(data, title=title).mark_circle(size=60).encode(
                x = col1,
                y = col2,
                color = color_var,
                tooltip = tooltip
            )
    return chart_dict

# TODO #9 tests for chart_cluster_compare
def chart_cluster_compare(data_array, xlabels, ylabels, x_lab, y_lab, z_lab, text_precision = ".0f"):
    """
    chart_cluster_compare Heatmap of the comparison percentage overlap of two clustering methods


    :param data_array: comparison data - long form - 
                    x_lab (clust_no. for method 1), y_lab (clust_no for method 2),
                    z_lab - no. in intersection/ no. in union
    :type data_array: numpy array
    :param xlabels: cluster labels for the columns 
    :type xlabels: list of strings
    :param ylabels: cluster labels for the rows
    :type ylabels: list of strings
    :param x_lab: label to get the x data from
    :type x_lab: string
    :param y_lab: label to get the y data from
    :type y_lab: string
    :param z_lab: label for the column containing the overlap percentage
    :type z_lab: string
    :param text_precision: precision for the printing of the numeric variables., defaults to ".0f"
    :type text_precision: str, optional
    """
    # Convert this grid to columnar data expected by Altair
    ylen = data_array.shape[0]
    xlen = data_array.shape[1]
    x, y = np.meshgrid(range(0, xlen), range(0, ylen))
    x_clust = {int(np.where(xlabels == cn)[0]) : cn for cn in xlabels}
    y_clust = {int(np.where(ylabels == cn)[0]) : cn for cn in ylabels}
    z = data_array
    source = pd.DataFrame({x_lab: x.ravel(),
                     y_lab: y.ravel(),
                     z_lab: z.ravel()})
    for i in range(0, xlen):
        source.loc[source[x_lab] == i, x_lab] = x_clust[i]
    for j in range(0, ylen):
        source.loc[source[y_lab] == j, y_lab] = y_clust[j]
    chart = alt.Chart(source).mark_rect().encode(
        alt.X(x_lab+":O").title(x_lab),
        alt.Y(y_lab+":O").title(y_lab),
        alt.Color(z_lab+":Q").title(z_lab)
        ).properties(
            width=400,
            height=400
        )
    text = chart.mark_text(baseline='middle').encode(
    alt.Text(z_lab+':Q', format = text_precision),
    color=alt.condition(
        alt.datum[z_lab]> 40,
        alt.value('white'),
        alt.value('black')
    )
)
    return(chart+text)